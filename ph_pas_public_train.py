"""
PH-PAS public training code
===========================

Publicly released components:
1. 3D ResNet-18 video encoder initialized with Kinetics-400 weights
2. Echocardiographic view embedding
3. Imaging-modality embedding
4. Masked-report-assisted video-text contrastive learning
5. Three-class classifier: Low / Intermediate / High PH probability

Not included:
- Structured clinical variables
- Patient-level multi-video aggregation code
- Private data-processing pipelines
- Patient data or model weights

Important
---------
Both --train_csv and --dev_csv must be derived from the model-development
training cohort only. Internal-test and external-development data, including
their echocardiographic reports, are not used by this training script.

Expected CSV columns
--------------------
VideoPath      : path to a video file
PH_label       : integer class label, 0=Low, 1=Intermediate, 2=High
View           : view label, e.g. PLAX / PSAX / A4C
Modality       : modality label, e.g. B-mode / CDFI
masked_report  : de-identified and label-masked echocardiographic report text

Example:
python ph_pas_public_train.py \
    --train_csv train.csv \
    --dev_csv dev.csv \
    --output_dir checkpoints
"""

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class VideoTextDataset(Dataset):
    """
    Each sample contains:
        video tensor
        PH probability label
        tokenized masked report
        view index
        modality index

    No structured clinical variables are used.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        num_frames: int = 20,
        tokenizer=None,
        view_to_idx=None,
        modality_to_idx=None,
    ):
        required_columns = {
            "VideoPath",
            "PH_label",
            "View",
            "Modality",
            "masked_report",
        }
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required CSV columns: {sorted(missing)}"
            )

        self.video_paths = df["VideoPath"].astype(str).tolist()
        self.labels = df["PH_label"].astype(int).tolist()
        self.views = df["View"].tolist()
        self.modalities = df["Modality"].tolist()
        self.masked_reports = (
            df["masked_report"].fillna("").astype(str).tolist()
        )

        self.transform = transform
        self.num_frames = num_frames
        self.tokenizer = tokenizer

        if view_to_idx is None or modality_to_idx is None:
            raise ValueError("view_to_idx and modality_to_idx are required.")

        self.view_to_idx = view_to_idx
        self.modality_to_idx = modality_to_idx

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, video_path: str):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames <= 0:
            raise ValueError(f"Video contains no frames: {video_path}")

        frame_indices = np.linspace(
            0, total_frames - 1, self.num_frames
        ).astype(int)

        frames = []
        for frame_idx in frame_indices:
            frame = vr[frame_idx].asnumpy()

            # Decord returns RGB arrays.
            frame = Image.fromarray(frame)

            if self.transform is not None:
                frame = self.transform(frame)

            frames.append(frame)

        # [C, T, H, W]
        return torch.stack(frames, dim=1)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        try:
            video = self._load_video(video_path)

            encoded = self.tokenizer(
                self.masked_reports[idx],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )

            label = torch.tensor(self.labels[idx], dtype=torch.long)
            view_idx = torch.tensor(
                self.view_to_idx[self.views[idx]],
                dtype=torch.long,
            )
            modality_idx = torch.tensor(
                self.modality_to_idx[self.modalities[idx]],
                dtype=torch.long,
            )

            return {
                "video": video,
                "label": label,
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "view": view_idx,
                "modality": modality_idx,
            }

        except Exception as exc:
            print(f"[Warning] Failed to process {video_path}: {exc}")
            return None


def collate_fn(batch):
    batch = [
        item
        for item in batch
        if item is not None
        and tuple(item["video"].shape) == (3, 20, 224, 224)
    ]

    if not batch:
        return None

    return default_collate(batch)


# ---------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------
class TextEncoder(nn.Module):
    """
    Masked echocardiographic reports are used only during training
    for video-text contrastive representation learning.
    """

    def __init__(
        self,
        pretrained_model_name: str = "hfl/chinese-bert-wwm-ext",
        projection_dim: int = 512,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.projection = nn.Linear(
            self.bert.config.hidden_size,
            projection_dim,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_feature = outputs.last_hidden_state[:, 0, :]
        text_feature = self.projection(cls_feature)

        return nn.functional.normalize(
            text_feature,
            dim=1,
        )


class VideoEncoder(nn.Module):
    """
    Video encoder + view embedding + modality embedding.

    Structured clinical variables are intentionally not used.
    """

    def __init__(
        self,
        num_views: int,
        num_modalities: int,
        feature_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.backbone = r3d_18(
            weights=R3D_18_Weights.KINETICS400_V1
        )
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            feature_dim,
        )

        self.view_embedding = nn.Embedding(
            num_views,
            feature_dim,
        )
        self.modality_embedding = nn.Embedding(
            num_modalities,
            feature_dim,
        )

        # 512 video + 512 view + 512 modality -> 512 fused representation
        self.fusion = nn.Linear(
            feature_dim * 3,
            feature_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, video, view, modality):
        video_feature = self.backbone(video)
        video_feature = nn.functional.normalize(
            video_feature,
            dim=1,
        )

        view_feature = self.view_embedding(view)
        modality_feature = self.modality_embedding(modality)

        fused_feature = torch.cat(
            [
                video_feature,
                view_feature,
                modality_feature,
            ],
            dim=1,
        )

        fused_feature = self.fusion(fused_feature)
        fused_feature = self.dropout(fused_feature)

        return nn.functional.normalize(
            fused_feature,
            dim=1,
        )


class PHPAS(nn.Module):
    """
    Training:
        video + view + modality + masked report

    Inference:
        video + view + modality only
    """

    def __init__(
        self,
        num_views: int,
        num_modalities: int,
        num_classes: int = 3,
        text_model_name: str = "hfl/chinese-bert-wwm-ext",
    ):
        super().__init__()

        self.video_encoder = VideoEncoder(
            num_views=num_views,
            num_modalities=num_modalities,
        )

        self.text_encoder = TextEncoder(
            pretrained_model_name=text_model_name,
        )

        self.classifier = nn.Linear(
            512,
            num_classes,
        )

    def forward(
        self,
        video,
        view,
        modality,
        input_ids=None,
        attention_mask=None,
    ):
        video_feature = self.video_encoder(
            video,
            view,
            modality,
        )

        logits = self.classifier(video_feature)

        text_feature = None
        if input_ids is not None and attention_mask is not None:
            text_feature = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return video_feature, text_feature, logits

    @torch.no_grad()
    def inference(self, video, view, modality):
        """
        Final inference path:
        no text and no structured clinical variables are required.
        """
        self.eval()

        video_feature = self.video_encoder(
            video,
            view,
            modality,
        )

        logits = self.classifier(video_feature)
        probabilities = torch.softmax(logits, dim=1)

        return probabilities


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
def symmetric_contrastive_loss(
    video_features,
    text_features,
    temperature: float = 0.07,
):
    """
    Symmetric video-to-text and text-to-video contrastive objective.
    """

    similarity = (
        video_features @ text_features.T
    ) / temperature

    targets = torch.arange(
        similarity.size(0),
        device=similarity.device,
    )

    loss_v2t = nn.CrossEntropyLoss()(
        similarity,
        targets,
    )
    loss_t2v = nn.CrossEntropyLoss()(
        similarity.T,
        targets,
    )

    return 0.5 * (loss_v2t + loss_t2v)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def run_epoch(
    model,
    dataloader,
    device,
    optimizer=None,
    beta: float = 3.74,
    temperature: float = 0.07,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(
        dataloader,
        desc="train" if is_training else "dev",
    ):
        if batch is None:
            continue

        video = batch["video"].to(device)
        labels = batch["label"].to(device)
        view = batch["view"].to(device)
        modality = batch["modality"].to(device)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            (
                video_features,
                text_features,
                logits,
            ) = model(
                video=video,
                view=view,
                modality=modality,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            contrastive_loss = symmetric_contrastive_loss(
                video_features,
                text_features,
                temperature=temperature,
            )

            classification_loss = nn.CrossEntropyLoss()(
                logits,
                labels,
            )

            # L_total = L_con + beta * L_cls
            # beta = 3.74 corresponds to the relative weighting
            # of the original 0.38 * L_con + 1.42 * L_cls formulation.
            loss = (
                contrastive_loss
                + beta * classification_loss
            )

            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()

    if total_samples == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
        }

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_model(
    model,
    train_loader,
    dev_loader,
    device,
    output_dir,
    epochs: int = 30,
    beta: float = 3.74,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_dev_loss = float("inf")
    best_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(epochs):
        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            beta=beta,
        )

        dev_metrics = run_epoch(
            model=model,
            dataloader=dev_loader,
            device=device,
            optimizer=None,
            beta=beta,
        )

        scheduler.step(
            dev_metrics["loss"]
        )

        print(
            "Train - "
            f"loss: {train_metrics['loss']:.4f}, "
            f"accuracy: {train_metrics['accuracy']:.4f}"
        )
        print(
            "Dev   - "
            f"loss: {dev_metrics['loss']:.4f}, "
            f"accuracy: {dev_metrics['accuracy']:.4f}"
        )

        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_state,
                output_dir / "ph_pas_best.pt",
            )

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def build_mapping(values):
    unique_values = sorted(
        pd.Series(values).dropna().unique().tolist()
    )
    return {
        value: idx
        for idx, value in enumerate(unique_values)
    }


def validate_label_range(df):
    labels = sorted(
        df["PH_label"].dropna().astype(int).unique()
    )

    if not set(labels).issubset({0, 1, 2}):
        raise ValueError(
            "PH_label must use "
            "0=Low, 1=Intermediate, 2=High."
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_csv",
        required=True,
    )
    parser.add_argument(
        "--dev_csv",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints",
    )
    parser.add_argument(
        "--text_model",
        default="hfl/chinese-bert-wwm-ext",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    train_df = pd.read_csv(
        args.train_csv,
        encoding="utf-8-sig",
    )
    dev_df = pd.read_csv(
        args.dev_csv,
        encoding="utf-8-sig",
    )

    validate_label_range(train_df)
    validate_label_range(dev_df)

    # Mapping is determined from training data only.
    view_to_idx = build_mapping(
        train_df["View"]
    )
    modality_to_idx = build_mapping(
        train_df["Modality"]
    )

    unseen_dev_views = set(
        dev_df["View"].dropna().unique()
    ) - set(view_to_idx)

    unseen_dev_modalities = set(
        dev_df["Modality"].dropna().unique()
    ) - set(modality_to_idx)

    if unseen_dev_views:
        raise ValueError(
            f"Development split contains unseen views: "
            f"{sorted(unseen_dev_views)}"
        )

    if unseen_dev_modalities:
        raise ValueError(
            f"Development split contains unseen modalities: "
            f"{sorted(unseen_dev_modalities)}"
        )

    tokenizer = BertTokenizer.from_pretrained(
        args.text_model
    )

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.43216,
                    0.394666,
                    0.37645,
                ],
                std=[
                    0.22803,
                    0.22145,
                    0.216989,
                ],
            ),
        ]
    )

    train_dataset = VideoTextDataset(
        train_df,
        transform=transform,
        num_frames=20,
        tokenizer=tokenizer,
        view_to_idx=view_to_idx,
        modality_to_idx=modality_to_idx,
    )

    dev_dataset = VideoTextDataset(
        dev_df,
        transform=transform,
        num_frames=20,
        tokenizer=tokenizer,
        view_to_idx=view_to_idx,
        modality_to_idx=modality_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = PHPAS(
        num_views=len(view_to_idx),
        num_modalities=len(modality_to_idx),
        num_classes=3,
        text_model_name=args.text_model,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    print("Class mapping:")
    print(
        "0 = Low, "
        "1 = Intermediate, "
        "2 = High"
    )
    print(
        f"Views: {view_to_idx}"
    )
    print(
        f"Modalities: {modality_to_idx}"
    )

    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        beta=3.74,
    )


if __name__ == "__main__":
    main()
