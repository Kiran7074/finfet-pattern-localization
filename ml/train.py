import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from ml.dataset import DriftSenseDataset
from ml.model import SiamXCorrNet, SoftArgmax2D


def main():
    # Configuration
    # File paths for train and val images from the dataset
    TRAIN_CSV = "./output_5000/train/manifest.csv"
    VAL_CSV = "./output_5000/val/manifest.csv"

    # Choosed according to the VRAM of the GPU
    BATCH_SIZE = 6
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    # Gaussian target width in heatmap coordinates
    GAUSSIAN_SIGMA = 1.5
    # Weight given to coordinate loss
    LAMBDA_COORD = 2.0
    MODEL_SAVE_PATH = "./model/siamese_cnn.pt"

    # Selection of the Device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Your ref/search image sizes never change (100x100 / 1000x1000),
    # so let cuDNN benchmark and cache the fastest conv algorithms for
    # those exact shapes instead of re-picking a generic one each call.
    torch.backends.cudnn.benchmark = True
    print("=" * 60)
    print("DEVICE")
    print("=" * 60)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    print()

    # Loading the Dataset

    train_dataset = DriftSenseDataset(TRAIN_CSV)
    val_dataset = DriftSenseDataset(VAL_CSV)
    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print()

    NUM_WORKERS = 0  # tune according to the CPU cores

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )

    # Model

    model = SiamXCorrNet().to(device)

    soft_argmax = SoftArgmax2D(
        temperature=20.0
    ).to(device)


    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(soft_argmax.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # Mixed precision: runs conv/matmul in fp16 on your 3050's tensor
    # cores while keeping a master fp32 copy of weights for stability.
    # Typically ~2x faster and uses noticeably less VRAM, which also
    scaler = GradScaler()
    with torch.no_grad():

        dummy_ref = torch.zeros(1, 1, 100, 100,device=device)
        dummy_search = torch.zeros(1, 1, 1000, 1000,device=device)
        # Multi-scale backbone
        ref_fine, ref_coarse = model.backbone(dummy_ref)
        search_fine, search_coarse = model.backbone(dummy_search)
        # Multi-scale cross correlation
        dummy_heatmap = model.xcorr(
            ref_fine,
            search_fine,
            ref_coarse,
            search_coarse
        )

    # Search feature -> original image scale

    STRIDE_X = 1000 / search_fine.shape[-1]
    STRIDE_Y = 1000 / search_fine.shape[-2]

    REF_HALF_X = ref_fine.shape[-1] / 2.0
    REF_HALF_Y = ref_fine.shape[-2] / 2.0


    print("=" * 60)
    print("COORDINATE GEOMETRY")
    print("=" * 60)

    print()
    print("REFERENCE FINE :", ref_fine.shape)
    print("SEARCH FINE    :", search_fine.shape)

    print("REFERENCE COARSE :", ref_coarse.shape)
    print("SEARCH COARSE    :", search_coarse.shape)

    print("HEATMAP :", dummy_heatmap.shape)

    print("STRIDE :", STRIDE_X)
    print("OFFSET :", REF_HALF_X)

    print("Stride X:", STRIDE_X)
    print("Stride Y:", STRIDE_Y)

    print("Offset X:", REF_HALF_X)
    print("Offset Y:", REF_HALF_Y)

    print()


    # Ground truth conversion

    def pixel_to_heatmap_coords(gt_x,gt_y):
        fx = (gt_x / STRIDE_X - REF_HALF_X)
        fy = (gt_y / STRIDE_Y- REF_HALF_Y)
        return fx, fy

    # Gaussian target
    def make_gaussian_target(gt_x,gt_y,H,W,sigma,device):
        B = gt_x.shape[0]
        xs = torch.arange(W,device=device,dtype=torch.float32)
        ys = torch.arange(H,device=device,dtype=torch.float32)

        grid_y, grid_x = torch.meshgrid(ys,xs,indexing="ij")
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand( B, -1, -1)
        gt_x = gt_x.view(B, 1, 1)
        gt_y = gt_y.view(B, 1, 1)
        gaussian = torch.exp(
            -(
                (grid_x - gt_x) ** 2 +
                (grid_y - gt_y) ** 2
            )
            /
            (2 * sigma ** 2)
        )
        gaussian = gaussian / gaussian.sum(dim=(1, 2),keepdim=True)
        return gaussian.unsqueeze(1)


    # Validation function

    def validate():
        model.eval()
        soft_argmax.eval()
        errors = []
        with torch.no_grad():
            for ref, search, gt in val_loader:
                ref = ref.to(device,non_blocking=True)
                search = search.to(device,non_blocking=True)
                gt = gt.to(device)
                with autocast():
                    # CNN
                    heatmap, _, _, _, _ = model(ref, search)
                    # Heatmap -> coordinate
                    pred_x, pred_y, _ = soft_argmax(heatmap)
                pred_x = pred_x.float()
                pred_y = pred_y.float()
                # Heatmap coordinate -> original pixel
                pred_pixel_x = (pred_x + REF_HALF_X) * STRIDE_X
                pred_pixel_y = (pred_y + REF_HALF_Y) * STRIDE_Y
                # Center localization error
                error = torch.sqrt((pred_pixel_x - gt[:, 0]) ** 2 +(pred_pixel_y - gt[:, 1]) ** 2)
                errors.extend(error.cpu().numpy())
        errors = np.array(errors)
        mean_cle = np.mean(errors)
        median_cle = np.median(errors)
        acc_1 = np.mean(errors <= 1) * 100
        acc_2 = np.mean(errors <= 2) * 100
        acc_5 = np.mean(errors <= 5) * 100
        acc_10 = np.mean(errors <= 10) * 100
        return (mean_cle,median_cle,acc_1,acc_2,acc_5,acc_10)


    # Training

    best_cle = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        soft_argmax.train()
        running_loss = 0.0
        for batch_idx, (ref, search, gt) in enumerate(
            train_loader
        ):
            ref = ref.to(
                device,
                non_blocking=True
            )

            search = search.to(
                device,
                non_blocking=True
            )
            gt = gt.to(device)
            optimizer.zero_grad()
            # Forward (mixed precision)
            with autocast():
                heatmap, _, _, _, _ = model(ref, search)
                pred_x, pred_y, probability = soft_argmax(
                    heatmap
                )
                # Softmax/coord math is sensitive to fp16 range, so do
                # it in fp32 even though the backbone/xcorr ran in fp16.
                pred_x = pred_x.float()
                pred_y = pred_y.float()
                probability = probability.float()
                # Convert GT from image pixels -> heatmap pixels
                gt_heat_x, gt_heat_y = pixel_to_heatmap_coords(
                    gt[:, 0],
                    gt[:, 1]
                )
                # Gaussian target
                H, W = heatmap.shape[-2:]
                target_probability = make_gaussian_target(
                    gt_heat_x,
                    gt_heat_y,
                    H,
                    W,
                    GAUSSIAN_SIGMA,
                    device
                )

                # Loss 1: heatmap
                heatmap_loss = F.mse_loss(
                    probability,
                    target_probability
                )
                # Loss 2: coordinate
                coordinate_loss = (
                    F.smooth_l1_loss(
                        pred_x,
                        gt_heat_x
                    )
                    +
                    F.smooth_l1_loss(
                        pred_y,
                        gt_heat_y
                    )
                )
                # Total loss
                loss = (
                    heatmap_loss
                    +
                    LAMBDA_COORD * coordinate_loss
                )

            # ----------------------------------------------------
            # Backpropagation (scaled to avoid fp16 underflow)
            # ----------------------------------------------------

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # Validation
        mean_cle, median_cle, acc_1, acc_2, acc_5, acc_10 = validate()
        scheduler.step(mean_cle)

        avg_loss = (running_loss /len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Mean CLE: {mean_cle:.2f}px | "
            f"Median CLE: {median_cle:.2f}px | "
            f"@1px: {acc_1:.1f}% | "
            f"@2px: {acc_2:.1f}% | "
            f"@5px: {acc_5:.1f}% | "
            f"@10px: {acc_10:.1f}% | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if mean_cle < best_cle:
            best_cle = mean_cle
            torch.save(
                {
                    "model": model.state_dict(),
                    "soft_argmax": soft_argmax.state_dict(),
                    "epoch": epoch + 1,
                    "mean_cle": mean_cle
                },
                MODEL_SAVE_PATH
            )

            print(
                f"  -> Saved best model "
                f"(CLE = {mean_cle:.2f}px)"
            )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation CLE: {best_cle:.2f}px"
    )

    print(
        f"Best model saved to: {MODEL_SAVE_PATH}"
    )

if __name__ == "__main__":
    main()