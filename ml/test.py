import torch
import numpy as np
from torch.utils.data import DataLoader

from ml.dataset import DriftSenseDataset
from ml.model import SiamXCorrNet, SoftArgmax2D


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "./output_5000/test/manifest.csv"
MODEL_PATH = "best_model_new_aug.pt"

BATCH_SIZE = 6
NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Dataset
# ============================================================

dataset = DriftSenseDataset(TEST_CSV)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Test samples:", len(dataset))


# ============================================================
# Model
# ============================================================

model = SiamXCorrNet().to(device)

soft_argmax = SoftArgmax2D(
    temperature=100.0
).to(device)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model"])
soft_argmax.load_state_dict(checkpoint["soft_argmax"])

print()
print("Loaded model successfully!")
print("Saved epoch:", checkpoint["epoch"])
print("Saved Test CLE:", checkpoint["mean_cle"])


# ============================================================
# Evaluation mode
# ============================================================

model.eval()
soft_argmax.eval()


# ============================================================
# Coordinate geometry
# ============================================================

with torch.no_grad():

    dummy_ref = torch.zeros(
        1, 1, 100, 100,
        device=device
    )

    dummy_search = torch.zeros(
        1, 1, 1000, 1000,
        device=device
    )

    # --------------------------------------------------------
    # NEW MULTI-SCALE BACKBONE
    # --------------------------------------------------------

    ref_fine, ref_coarse = model.backbone(
        dummy_ref
    )

    search_fine, search_coarse = model.backbone(
        dummy_search
    )

    # --------------------------------------------------------
    # NEW MULTI-SCALE XCORR
    # --------------------------------------------------------

    dummy_heatmap = model.xcorr(
        ref_fine,
        search_fine,
        ref_coarse,
        search_coarse
    )


# ============================================================
# Fine feature geometry
# ============================================================

STRIDE_X = 1000 / search_fine.shape[-1]
STRIDE_Y = 1000 / search_fine.shape[-2]

REF_HALF_X = ref_fine.shape[-1] / 2.0
REF_HALF_Y = ref_fine.shape[-2] / 2.0


# ============================================================
# Print dimensions
# ============================================================

print()

print("Reference fine:", ref_fine.shape)
print("Search fine:", search_fine.shape)

print("Reference coarse:", ref_coarse.shape)
print("Search coarse:", search_coarse.shape)

print("Heatmap:", dummy_heatmap.shape)

print("Stride X:", STRIDE_X)
print("Stride Y:", STRIDE_Y)

print("Offset X:", REF_HALF_X)
print("Offset Y:", REF_HALF_Y)


# ============================================================
# Storage
# ============================================================

errors = []

predictions = []

ground_truths = []


# ============================================================
# Testing
# ============================================================

sample_number = 0

with torch.no_grad():

    for ref, search, gt in loader:

        ref = ref.to(
            device,
            non_blocking=True
        )

        search = search.to(
            device,
            non_blocking=True
        )

        gt = gt.to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        heatmap, _, _, _, _ = model(
            ref,
            search
        )


        # ----------------------------------------------------
        # Heatmap -> heatmap coordinates
        # ----------------------------------------------------

        pred_x, pred_y, probability = soft_argmax(
            heatmap
        )

        pred_x = pred_x.float()
        pred_y = pred_y.float()


        # ----------------------------------------------------
        # Heatmap coordinates -> original image coordinates
        # ----------------------------------------------------

        pred_pixel_x = (
            pred_x + REF_HALF_X
        ) * STRIDE_X

        pred_pixel_y = (
            pred_y + REF_HALF_Y
        ) * STRIDE_Y


        # ----------------------------------------------------
        # Calculate localization error
        # ----------------------------------------------------

        error = torch.sqrt(
            (pred_pixel_x - gt[:, 0]) ** 2 +
            (pred_pixel_y - gt[:, 1]) ** 2
        )


        # ----------------------------------------------------
        # Print every sample
        # ----------------------------------------------------

        for i in range(len(gt)):

            sample_number += 1

            gt_x = gt[i, 0].item()
            gt_y = gt[i, 1].item()

            prediction_x = pred_pixel_x[i].item()
            prediction_y = pred_pixel_y[i].item()

            sample_error = error[i].item()

            print(
                f"Sample {sample_number:02d} | "
                f"GT: ({gt_x:.2f}, {gt_y:.2f}) | "
                f"Pred: ({prediction_x:.2f}, {prediction_y:.2f}) | "
                f"Error: {sample_error:.2f}px"
            )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        errors.extend(
            error.cpu().numpy()
        )

        predictions.extend(
            torch.stack(
                [
                    pred_pixel_x,
                    pred_pixel_y
                ],
                dim=1
            ).cpu().numpy()
        )

        ground_truths.extend(
            gt.cpu().numpy()
        )


# ============================================================
# Convert to NumPy
# ============================================================

errors = np.array(errors)
# ============================================================
# SAVE MODEL ERRORS
# ============================================================


predictions = np.array(predictions)

ground_truths = np.array(ground_truths)

np.savetxt(
    "model_errors.csv",
    errors,
    delimiter=",",
    header="error_px",
    comments=""
)

print("\nSaved model errors to model_errors.csv")

# ============================================================
# Metrics
# ============================================================

mean_cle = np.mean(errors)

median_cle = np.median(errors)

acc_1 = np.mean(errors <= 1) * 100

acc_2 = np.mean(errors <= 2) * 100

acc_5 = np.mean(errors <= 5) * 100

acc_10 = np.mean(errors <= 10) * 100


# ============================================================
# Find best and worst samples
# ============================================================

best_index = np.argmin(errors)

worst_index = np.argmax(errors)


# ============================================================
# Results
# ============================================================

print()

print("=" * 70)

print("TEST RESULTS")

print("=" * 70)

print(f"Number of samples: {len(errors)}")

print()

print(f"Mean CLE:   {mean_cle:.2f} px")

print(f"Median CLE: {median_cle:.2f} px")

print()

print(f"@1 px:      {acc_1:.2f}%")

print(f"@2 px:      {acc_2:.2f}%")

print(f"@5 px:      {acc_5:.2f}%")

print(f"@10 px:     {acc_10:.2f}%")

print()

print("=" * 70)

print("BEST PREDICTION")

print("=" * 70)

print(
    f"GT   : ({ground_truths[best_index][0]:.2f}, "
    f"{ground_truths[best_index][1]:.2f})"
)

print(
    f"Pred : ({predictions[best_index][0]:.2f}, "
    f"{predictions[best_index][1]:.2f})"
)

print(
    f"Error: {errors[best_index]:.2f} px"
)

print()

print("=" * 70)

print("WORST PREDICTION")

print("=" * 70)

print(
    f"GT   : ({ground_truths[worst_index][0]:.2f}, "
    f"{ground_truths[worst_index][1]:.2f})"
)

print(
    f"Pred : ({predictions[worst_index][0]:.2f}, "
    f"{predictions[worst_index][1]:.2f})"
)

print(
    f"Error: {errors[worst_index]:.2f} px"
)

print("=" * 70)