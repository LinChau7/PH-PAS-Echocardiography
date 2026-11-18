"""Generate class-specific 3D Grad-CAM maps for the public PH-PAS model.

This script is matched to the public ``ph_pas_public_train.py`` architecture.
It explains a VIDEO-LEVEL class logit while holding the view and modality
embedding inputs fixed. The text encoder is not needed for inference and is
therefore omitted when loading the checkpoint.

Expected model inputs:
    video:    (1, 3, 20, 224, 224)
    view:     one integer embedding index used during training
    modality: one integer embedding index used during training

The script accepts only the public model checkpoint without structured
clinical variables and does not read DICOM files. Use de-identified AVI/MP4
videos and the same preprocessing as the public training script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.video import r3d_18


CLASSES = ["Low", "Intermediate", "High"]
NUM_FRAMES = 20
FRAME_SIZE = 224
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]


class PublicVideoEncoder(nn.Module):
    """Video, view, and modality encoder from public PH-PAS training code."""

    def __init__(self, num_views: int, num_modalities: int, feature_dim: int = 512):
        super().__init__()

        self.backbone = r3d_18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)

        self.view_embedding = nn.Embedding(num_views, feature_dim)
        self.modality_embedding = nn.Embedding(num_modalities, feature_dim)

        self.fusion = nn.Linear(feature_dim * 3, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, video: torch.Tensor, view: torch.Tensor, modality: torch.Tensor):
        video_feature = self.backbone(video)
        video_feature = F.normalize(video_feature, dim=1)

        view_feature = self.view_embedding(view)
        modality_feature = self.modality_embedding(modality)

        fused_feature = torch.cat(
            [video_feature, view_feature, modality_feature],
            dim=1,
        )
        fused_feature = self.fusion(fused_feature)
        fused_feature = self.dropout(fused_feature)

        return F.normalize(fused_feature, dim=1)


class PublicPHPASInference(nn.Module):
    """Inference-only PH-PAS model; the training-only text branch is omitted."""

    def __init__(self, num_views: int, num_modalities: int, num_classes: int = 3):
        super().__init__()
        self.video_encoder = PublicVideoEncoder(
            num_views=num_views,
            num_modalities=num_modalities,
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, video: torch.Tensor, view: torch.Tensor, modality: torch.Tensor):
        features = self.video_encoder(video, view, modality)
        return self.classifier(features)


def _torch_load_state(path: Path):
    """Load tensor weights safely where the installed PyTorch supports it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with PyTorch versions predating weights_only
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint_obj) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint_obj, Mapping):
        raise TypeError("Checkpoint must be a state_dict or a dictionary containing one.")

    state = checkpoint_obj
    for wrapper_key in ("state_dict", "model_state_dict"):
        wrapped = state.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            state = wrapped
            break

    tensor_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        # The BERT text branch is used only for training-time contrastive learning.
        if clean_key.startswith("text_encoder."):
            continue
        tensor_state[clean_key] = value

    if not tensor_state:
        raise ValueError("No model tensors were found in the checkpoint.")
    return tensor_state


def load_public_model(checkpoint_path: Path, device: torch.device):
    """Load only a checkpoint matching the public no-clinical PH-PAS model."""
    checkpoint_obj = _torch_load_state(checkpoint_path)
    state = _extract_state_dict(checkpoint_obj)

    required_keys = (
        "video_encoder.backbone.fc.weight",
        "video_encoder.backbone.fc.bias",
        "video_encoder.view_embedding.weight",
        "video_encoder.modality_embedding.weight",
        "video_encoder.fusion.weight",
        "video_encoder.fusion.bias",
        "classifier.weight",
        "classifier.bias",
    )
    missing = [key for key in required_keys if key not in state]
    if missing:
        raise ValueError(
            "Checkpoint does not match the public PH-PAS architecture. "
            f"Missing required parameter(s): {missing}"
        )

    num_views, view_dim = state["video_encoder.view_embedding.weight"].shape
    num_modalities, modality_dim = state["video_encoder.modality_embedding.weight"].shape
    if view_dim != 512 or modality_dim != 512:
        raise ValueError("Expected 512-dimensional view and modality embeddings.")
    if tuple(state["video_encoder.fusion.weight"].shape) != (512, 1536):
        raise ValueError("Expected the public PH-PAS fusion layer to have shape (512, 1536).")
    if tuple(state["video_encoder.backbone.fc.weight"].shape) != (512, 512):
        raise ValueError("Expected the 3D ResNet-18 projection layer to have shape (512, 512).")
    if tuple(state["classifier.weight"].shape) != (3, 512):
        raise ValueError("Expected a three-class classifier with weight shape (3, 512).")

    model = PublicPHPASInference(
        num_views=int(num_views),
        num_modalities=int(num_modalities),
        num_classes=3,
    )

    # Strict loading is intentional: do not silently ignore architecture mismatches.
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, int(num_views), int(num_modalities)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_video(path: Path):
    """Uniformly sample 20 RGB frames and apply the exact public-train transform."""
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError(
            "Decord is required to read videos. Install it with: pip install decord"
        ) from exc

    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames < 1:
        raise ValueError(f"Video contains no decodable frames: {path}")

    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)

    spatial_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((FRAME_SIZE, FRAME_SIZE)),
        ]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    processed_rgb = []
    frame_tensors = []
    for frame_index in frame_indices:
        # Decord returns RGB frames; the public training script passes them
        # directly to PIL.Image.fromarray without a BGR/RGB channel swap.
        pil_frame = Image.fromarray(reader[int(frame_index)].asnumpy())
        pil_frame = spatial_transform(pil_frame)
        processed_rgb.append(np.asarray(pil_frame, dtype=np.uint8))
        frame_tensors.append(tensor_transform(pil_frame))

    # [T,C,H,W] -> [1,C,T,H,W], matching torchvision R3D-18 input.
    video_tensor = torch.stack(frame_tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    if tuple(video_tensor.shape) != (1, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE):
        raise RuntimeError(f"Unexpected video tensor shape: {tuple(video_tensor.shape)}")

    return video_tensor.contiguous(), np.stack(processed_rgb), frame_indices


def compute_grad_cam(
    model: PublicPHPASInference,
    video: torch.Tensor,
    view: torch.Tensor,
    modality: torch.Tensor,
    class_index: int | None = None,
):
    """Return logits, probabilities, class indices, and raw/upsampled 3D CAMs."""
    if video.shape[0] != 1:
        raise ValueError("Grad-CAM is designed to process one video at a time.")

    target_layer = model.video_encoder.backbone.layer4
    captured_activations = []

    def save_activation(_module, _inputs, output):
        captured_activations.append(output)

    hook_handle = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(video, view, modality)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits.")

            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            target_class = predicted_class if class_index is None else int(class_index)
            if target_class not in (0, 1, 2):
                raise ValueError("class_index must be 0, 1, or 2.")
            if not captured_activations:
                raise RuntimeError("Could not capture activations from backbone.layer4.")

            activation = captured_activations[-1]
            gradient = torch.autograd.grad(
                outputs=logits[0, target_class],
                inputs=activation,
                retain_graph=False,
                create_graph=False,
            )[0]

            # Average gradients over temporal and spatial dimensions.
            channel_weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
            raw_cam_tensor = F.relu((channel_weights * activation).sum(dim=1, keepdim=True))
            raw_cam = raw_cam_tensor[0, 0].detach().cpu().numpy()

            # Upsample the feature-map CAM to the 20 x 224 x 224 input grid.
            upsampled = F.interpolate(
                raw_cam_tensor,
                size=video.shape[2:],
                mode="trilinear",
                align_corners=False,
            )[0, 0]
            cam_max_before_normalization = float(upsampled.max().detach().cpu())
            if cam_max_before_normalization > 0:
                normalized_cam = upsampled / cam_max_before_normalization
            else:
                normalized_cam = torch.zeros_like(upsampled)

            return {
                "logits": logits.detach()[0].cpu().numpy(),
                "probabilities": probabilities.detach()[0].cpu().numpy(),
                "predicted_class_index": predicted_class,
                "target_class_index": target_class,
                "raw_cam": raw_cam,
                "cam": normalized_cam.detach().cpu().numpy(),
                "raw_feature_cam_max": float(raw_cam.max()) if raw_cam.size else 0.0,
                "cam_max_before_normalization": cam_max_before_normalization,
            }
    finally:
        hook_handle.remove()


def heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    """Convert a [0,1] CAM to an RGB JET-like heatmap without extra dependencies."""
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def save_side_by_side_mp4(
    output_path: Path,
    original_frames: np.ndarray,
    overlay_images: list[Image.Image],
    fps: float = 10.0,
) -> bool:
    """Save an optional original/overlay MP4 when OpenCV is installed."""
    try:
        import cv2
    except ImportError:
        return False

    height, width = original_frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        writer.release()
        return False

    try:
        for original, overlay_image in zip(original_frames, overlay_images):
            overlay = np.asarray(overlay_image, dtype=np.uint8)
            side_by_side = np.concatenate([original, overlay], axis=1)
            writer.write(cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path.is_file() and output_path.stat().st_size > 0


def save_outputs(
    output_dir: Path,
    video_path: Path,
    processed_rgb: np.ndarray,
    sampled_frame_indices: np.ndarray,
    result: dict,
    checkpoint_path: Path,
    view_index: int,
    modality_index: int,
    alpha: float,
):
    """Save frame overlays, GIF/optional MP4 previews, CAM arrays, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=False)

    cam = result["cam"]
    overlays = []
    for frame_idx, (frame, heat) in enumerate(zip(processed_rgb, cam)):
        heat_rgb = heatmap_rgb(heat)
        opacity = (alpha * heat)[..., None]
        blended = (1.0 - opacity) * frame.astype(np.float32) + opacity * (heat_rgb * 255.0)
        overlay = np.clip(blended, 0, 255).astype(np.uint8)

        Image.fromarray(frame).save(output_dir / f"frame_{frame_idx:02d}.png")
        Image.fromarray(overlay).save(output_dir / f"overlay_{frame_idx:02d}.png")
        overlays.append(Image.fromarray(overlay))

    overlays[0].save(
        output_dir / "preview.gif",
        save_all=True,
        append_images=overlays[1:],
        duration=100,
        loop=0,
    )
    mp4_preview_saved = save_side_by_side_mp4(
        output_dir / "gradcam_overlay.mp4",
        processed_rgb,
        overlays,
    )

    np.save(output_dir / "cam.npy", cam)
    np.save(output_dir / "raw_cam.npy", result["raw_cam"])
    np.savez_compressed(
        output_dir / "cam_data.npz",
        sampled_frames_rgb=processed_rgb,
        cam_upsampled=cam,
        cam_feature_map=result["raw_cam"],
        sampled_frame_indices=sampled_frame_indices,
        logits=result["logits"],
        probabilities=result["probabilities"],
    )

    probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(CLASSES, result["probabilities"])
    }
    metadata = {
        "model_architecture": "public_ph_pas_no_structured_clinical_variables",
        "explanation_level": "video_level",
        "target_layer": "video_encoder.backbone.layer4",
        "predicted_class_index": result["predicted_class_index"],
        "predicted_class": CLASSES[result["predicted_class_index"]],
        "target_class_index_visualized": result["target_class_index"],
        "target_class_visualized": CLASSES[result["target_class_index"]],
        "logits": [float(x) for x in result["logits"]],
        "probabilities": probabilities,
        "view_embedding_index": int(view_index),
        "modality_embedding_index": int(modality_index),
        "video_filename": video_path.name,
        "video_sha256": sha256_file(video_path),
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "num_sampled_frames": NUM_FRAMES,
        "sampled_frame_indices_zero_based": [int(x) for x in sampled_frame_indices],
        "input_frame_size": [FRAME_SIZE, FRAME_SIZE],
        "preprocessing": [
            "Decord RGB decoding",
            "uniform sampling of 20 frames across the full video",
            "Resize((256, 256))",
            "CenterCrop((224, 224))",
            "ToTensor()",
            "Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])",
        ],
        "cam_method": "3D Grad-CAM; channel weights are gradients averaged over temporal and spatial feature-map dimensions",
        "cam_postprocessing": "ReLU, trilinear upsampling to 20x224x224, division by the maximum over the full clip; no min-max subtraction",
        "raw_feature_cam_max": result["raw_feature_cam_max"],
        "cam_max_before_normalization": result["cam_max_before_normalization"],
        "overlay_alpha": float(alpha),
        "gif_frame_duration_ms": 100,
        "mp4_preview_saved": bool(mp4_preview_saved),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "python_version": platform.python_version(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video-level 3D Grad-CAM for the public PH-PAS model."
    )
    parser.add_argument("--checkpoint", required=True, help="Public PH-PAS checkpoint (.pt/.pth)")
    parser.add_argument("--video", required=True, help="De-identified AVI/MP4 video used for model input")
    parser.add_argument(
        "--view-index",
        required=True,
        type=int,
        help="View embedding index from the training-time view_to_idx mapping",
    )
    parser.add_argument(
        "--modality-index",
        required=True,
        type=int,
        help="Modality embedding index from the training-time modality_to_idx mapping",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Class to visualize: 0=Low, 1=Intermediate, 2=High; default is predicted class",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type=float, default=0.40, help="Heatmap overlay opacity (0 to 1)")
    parser.add_argument("--output-dir", required=True, help="New output directory; it must not already exist")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new directory to avoid overwriting results."
        )
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    device = torch.device(args.device)
    model, num_views, num_modalities = load_public_model(checkpoint_path, device)

    if not 0 <= args.view_index < num_views:
        raise ValueError(
            f"view-index {args.view_index} is out of range; checkpoint has {num_views} view embeddings."
        )
    if not 0 <= args.modality_index < num_modalities:
        raise ValueError(
            f"modality-index {args.modality_index} is out of range; checkpoint has {num_modalities} modality embeddings."
        )

    video_tensor, processed_rgb, frame_indices = load_video(video_path)
    view = torch.tensor([args.view_index], dtype=torch.long, device=device)
    modality = torch.tensor([args.modality_index], dtype=torch.long, device=device)

    result = compute_grad_cam(
        model=model,
        video=video_tensor.to(device),
        view=view,
        modality=modality,
        class_index=args.class_index,
    )

    save_outputs(
        output_dir=output_dir,
        video_path=video_path,
        processed_rgb=processed_rgb,
        sampled_frame_indices=frame_indices,
        result=result,
        checkpoint_path=checkpoint_path,
        view_index=args.view_index,
        modality_index=args.modality_index,
        alpha=args.alpha,
    )

    predicted = CLASSES[result["predicted_class_index"]]
    target = CLASSES[result["target_class_index"]]
    print(f"Predicted class: {predicted}")
    print(f"Grad-CAM target: {target}")
    print(f"Probabilities: {result['probabilities'].round(4).tolist()}")
    print(f"Grad-CAM outputs saved to: {output_dir}")
    if result["cam_max_before_normalization"] == 0:
        print("Warning: the target class produced no positive Grad-CAM activation; saved CAM arrays are zero.")


if __name__ == "__main__":
    main()
