import argparse
import cv2
import torch

from ml.model import SiamXCorrNet, SoftArgmax2D


# Configuration

MODEL_PATH = "./model/siamese_cnn.pt"


# Command-line arguments

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run FinFET/DRAM pattern localization on one image pair."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference image"
    )

    parser.add_argument(
        "--search",
        required=True,
        help="Path to search image"
    )

    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to trained model checkpoint"
    )

    return parser.parse_args()


# Main

def main():

    args = parse_args()
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("=" * 60)
    print("DRIFT-SENSE INFERENCE")
    print("=" * 60)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    # Load images
    reference = cv2.imread(
        args.reference,
        cv2.IMREAD_GRAYSCALE
    )
    search = cv2.imread(
        args.search,
        cv2.IMREAD_GRAYSCALE
    )
    if reference is None:
        raise RuntimeError(
            f"Could not read reference image: {args.reference}"
        )
    if search is None:
        raise RuntimeError(
            f"Could not read search image: {args.search}"
        )
    print()
    print("Reference:", args.reference)
    print("Search   :", args.search)
    print()
    print("Original reference shape:", reference.shape)
    print("Original search shape   :", search.shape)

    # Reference preprocessing

    reference = cv2.resize(
        reference,
        (100, 100),
        interpolation=cv2.INTER_AREA
    )

    reference = reference.astype("float32") / 255.0
    search = search.astype("float32") / 255.0

    reference = torch.from_numpy(
        reference
    ).unsqueeze(0).unsqueeze(0)

    search = torch.from_numpy(
        search
    ).unsqueeze(0).unsqueeze(0)

    reference = reference.to(device)
    search = search.to(device)

    print()
    print("Model reference shape:", reference.shape)
    print("Model search shape   :", search.shape)

    # Load model

    model = SiamXCorrNet().to(device)

    soft_argmax = SoftArgmax2D(
        temperature=100.0
    ).to(device)

    checkpoint = torch.load(
        args.model,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    soft_argmax.load_state_dict(
        checkpoint["soft_argmax"]
    )

    model.eval()
    soft_argmax.eval()

    print()
    print("Model loaded successfully.")

    # Determine coordinate geometry

    with torch.no_grad():

        dummy_ref = torch.zeros(
            1,
            1,
            100,
            100,
            device=device
        )

        dummy_search = torch.zeros(
            1,
            1,
            search.shape[-2],
            search.shape[-1],
            device=device
        )

        ref_fine, ref_coarse = model.backbone(
            dummy_ref
        )

        search_fine, search_coarse = model.backbone(
            dummy_search
        )

        dummy_heatmap = model.xcorr(
            ref_fine,
            search_fine,
            ref_coarse,
            search_coarse
        )

    stride_x = search.shape[-1] / search_fine.shape[-1]
    stride_y = search.shape[-2] / search_fine.shape[-2]

    ref_half_x = ref_fine.shape[-1] / 2.0
    ref_half_y = ref_fine.shape[-2] / 2.0

    # Inference

    with torch.no_grad():

        heatmap, _, _, _, _ = model(
            reference,
            search
        )

        pred_x, pred_y, probability = soft_argmax(
            heatmap
        )

        pred_pixel_x = (
            pred_x + ref_half_x
        ) * stride_x

        pred_pixel_y = (
            pred_y + ref_half_y
        ) * stride_y

    # Results

    x = pred_pixel_x[0].item()
    y = pred_pixel_y[0].item()

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(f"({x:.2f},{y:.2f})")

    print()
    print("Localization complete.")


if __name__ == "__main__":
    main()