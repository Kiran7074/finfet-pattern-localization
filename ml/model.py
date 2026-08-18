import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out

class Backbone(nn.Module):
    """
    Multi-scale Siamese backbone.
    Input:
        [B, 1, H, W]
    Outputs:
        fine:
            stride 4
            e.g. 100x100 -> 25x25
        coarse:
            stride 8
            e.g. 100x100 -> 13x13
    """
    def __init__(self, channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            # Takes the greyscale image as the input and outputs 64 features
            nn.Conv2d(1,32,kernel_size=5,stride=2,padding=2,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64,kernel_size=3,stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Fine feature processing (for high resolution details)
        # Output of stem is served as the input 
        self.fine_blocks = nn.Sequential(
            nn.Conv2d(64,channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )        
        # Coarse feature processing (It is for low resolution)
        # This is added beacuse the model will look too closely on the finer details on feature processing
        # So there is a chance for loosing the genral details 
        self.downsample = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.coarse_blocks = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
        )
        # Normalization
        # If the batch size is too small then there is a higher chnace of batch norm to fail or becoming inaccurate
        # Groupnorm solves this issue by dividing 128 channel into 16 groups and normailizing across those groups
        self.fine_norm = nn.GroupNorm(16,channels)
        self.coarse_norm = nn.GroupNorm(16,channels)

    def forward(self, x):
        x = self.stem(x)
        # 100 -> 25
        fine = self.fine_blocks(x)
        # 25 -> 13
        coarse = self.downsample(fine)
        coarse = self.coarse_blocks(coarse)

        fine = self.fine_norm(fine)
        coarse = self.coarse_norm(coarse)

        # L2 normalization (making the mag of the feature vector into 1)
        fine = F.normalize(fine,p=2,dim=1)
        coarse = F.normalize(coarse,p=2,dim=1)
        return fine, coarse

# Cross Correlation
class XCorr(nn.Module):
    def forward(
        self,
        ref_feat,
        search_feat
    ):
        # Hr & Wr represents the Height and Width of the small reference template
        B, C, Hr, Wr = ref_feat.shape
        outputs = []
        for b in range(B):
            kernel = ref_feat[b].unsqueeze(0)
            search = search_feat[b].unsqueeze(0)
            # Calculating the dot product at every single pixel location
            response = F.conv2d(search,kernel)
            # normalize by number of feature elements
            response = response / (C * Hr * Wr) ** 0.5
            # Processed heatmap into a list
            outputs.append(response)
        return torch.cat(outputs,dim=0)

# Multi-scale correlation
class MultiScaleXCorr(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine_xcorr = XCorr()
        self.coarse_xcorr = XCorr()
        # to learn how much each scale contributes
        self.fine_weight = nn.Parameter(torch.tensor(1.0))
        self.coarse_weight = nn.Parameter(torch.tensor(1.0))
    def forward(
        self,
        ref_fine,
        search_fine,
        ref_coarse,
        search_coarse
    ):
        fine_response = self.fine_xcorr(ref_fine,search_fine)
        coarse_response = self.coarse_xcorr(ref_coarse,search_coarse)
        # Matching coarse response size to fine response size
        coarse_response = F.interpolate(coarse_response, size=fine_response.shape[-2:],mode="bilinear",align_corners=False)
        response = (self.fine_weight * fine_response + self.coarse_weight * coarse_response)
        return response
    
# Soft Argmax
# To find the exact coordinate from the heatmap
class SoftArgmax2D(nn.Module):
    def __init__(self, temperature=20.0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(
                torch.tensor(float(temperature))
            )
        )
    def forward(self, heatmap):
        B, _, H, W = heatmap.shape
        temperature = torch.exp(self.log_temperature).clamp( min=1.0,max=200.0)
        flat = heatmap.view(B,-1)
        # Per-sample standardization
        flat = (flat - flat.mean(dim=1, keepdim=True)) / (flat.std(dim=1, keepdim=True)+ 1e-6)
        flat = flat * temperature
        # COnverting the raw scores onto probability distribution
        prob = F.softmax(flat,dim=1)
        prob = prob.view(B,1,H,W)
        xs = torch.arange(W,device=heatmap.device,dtype=heatmap.dtype)
        ys = torch.arange(H,device=heatmap.device,dtype=heatmap.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        x = torch.sum(prob.squeeze(1) * grid_x,dim=(1, 2))
        y = torch.sum(prob.squeeze(1) * grid_y,dim=(1, 2))
        return x, y, prob

# Full Model

class SiamXCorrNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(channels=128)
        self.xcorr = MultiScaleXCorr()
    # Siamese architecture
    def forward(self,ref,search):
        ref_fine, ref_coarse = self.backbone(ref)
        search_fine, search_coarse = self.backbone(search)
        # combined heatmap indicating where the object is most likely present
        heatmap = self.xcorr(ref_fine,search_fine,ref_coarse,search_coarse)
        return (heatmap,ref_fine,search_fine,ref_coarse,search_coarse)
