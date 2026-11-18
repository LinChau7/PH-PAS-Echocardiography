# PH-PAS-Echocardiography

## AI-driven Pulmonary Hypertension Probability Assessment using Multiview Echocardiography

This repository contains the PyTorch implementation and publicly releasable inference components of PH-PAS, an AI framework for guideline-based pulmonary hypertension (PH) probability assessment from multiview transthoracic echocardiography (TTE) videos.

## 📋 Overview

PH-PAS integrates:

* **Echocardiographic Videos:** Spatio-temporal feature extraction using 3D ResNet-18.
* **View and Modality Embeddings:** Echocardiographic view and imaging modality information are incorporated into the video representation.
* **Vision-Language Pretraining:** Masked echocardiographic reports are used during training for video-text contrastive learning.
* **Multivideo Aggregation:** Video-level predictions can be aggregated at the examination level using confidence-weighted view and modality performance weights.

During inference, the model requires only TTE videos together with view and modality information. The text encoder is used only during training and is not required for inference.

## ⚙️ Project Structure

```text
├── ph_pas_public_train.py
├── patient_aggregation.py
├── grad_cam.py
├── requirements.txt
├── .gitignore
└── README.md
```

### `ph_pas_public_train.py`

Public implementation of the PH-PAS model architecture and training procedure, including the 3D video encoder, view and modality embeddings, masked-report-assisted video-text contrastive learning, and three-class PH probability classification.

### `patient_aggregation.py`

Code for examination-level aggregation of multiple video-level PH probability predictions.

Video-level class probabilities are aggregated using confidence-weighted view and modality performance weights. The aggregation weights are derived from the development subset and fixed before evaluation on the internal test and external validation cohorts.

This script does not use patient-level clinical variables or evaluation-set labels to determine the aggregation weights.

### `grad_cam.py`

Code for generating video-level 3D Grad-CAM visualizations for the public PH-PAS model.

The implementation uses the final convolutional block of the 3D ResNet-18 backbone to visualize class-specific model activation patterns while keeping the view and modality information fixed.

The script accepts de-identified echocardiographic AVI/MP4 videos and does not require DICOM files or structured clinical variables.

## 🛠️ Installation

### Prerequisites

* Python 3.8+
* CUDA-enabled GPU recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries include:

* `torch`, `torchvision`
* `decord`
* `transformers`
* `numpy`, `pandas`
* `scikit-learn`
* `Pillow`
* `tqdm`

## 📊 Data Preparation

The training script requires CSV files linking echocardiographic videos to PH probability labels and masked echocardiographic reports.

### CSV Format

| Column | Description |
|---|---|
| `VideoPath` | Path to the echocardiographic video |
| `PH_label` | 0 = Low, 1 = Intermediate, 2 = High |
| `View` | Echocardiographic view |
| `Modality` | Imaging modality |
| `masked_report` | De-identified and masked echocardiographic report |

The model uses standard multiview TTE videos, including:

* Parasternal long-axis (PLAX)
* Parasternal short-axis (PSAX)
* Apical four-chamber (A4C)

Imaging modalities include:

* B-mode
* Colour Doppler flow imaging (CDFI)

Each video is uniformly sampled to 20 frames and processed using the spatial preprocessing specified in the public model implementation.

## 🚀 Usage

### Training

```bash
python ph_pas_public_train.py \
  --train_csv train.csv \
  --dev_csv dev.csv \
  --output_dir checkpoints
```

Optional arguments include:

```text
--text_model
--epochs
--batch_size
--num_workers
--seed
```

The default text encoder is:

```text
hfl/chinese-bert-wwm-ext
```

### Examination-level aggregation

`patient_aggregation.py` can be used to aggregate multiple video-level predictions from the same TTE examination.

The aggregation uses fixed view- and modality-specific performance weights derived from the development subset. These weights should be determined before applying the aggregation to internal test or external validation data.

### Grad-CAM visualization

`grad_cam.py` generates class-specific video-level 3D Grad-CAM maps from de-identified echocardiographic videos.

The visualization uses the same video preprocessing and model architecture as the public PH-PAS implementation.

## 🧠 Model Architecture

The framework consists of two main encoders:

### Video Encoder

A 3D ResNet-18 pretrained on Kinetics-400 processes 20-frame TTE video clips to extract spatio-temporal features.

Echocardiographic **view embeddings** and **modality embeddings** are combined with the video representation before classification.

### Text Encoder

A BERT model encodes de-identified and masked echocardiographic reports.

The text encoder is used only during training to align video representations with semantic information through video-text contrastive learning.

### Classification

The model predicts three echocardiographic PH probability categories:

* Low
* Intermediate
* High

The training objective is:

```text
L_total = L_con + 3.74 × L_cls
```

where `L_con` denotes the symmetric video-text contrastive loss and `L_cls` denotes the three-class cross-entropy classification loss.

The contrastive temperature is set to `0.07`.

### Inference

During inference, the text encoder is removed.

PH-PAS uses only:

* TTE videos
* View information
* Modality information

## 🔎 Interpretability

The publicly released `grad_cam.py` script provides video-level 3D Grad-CAM visualization for model interpretation.

The visualization identifies spatio-temporal regions contributing to a selected PH probability class. It is intended for qualitative model interpretation and does not represent an independent diagnostic measurement.

## 🔐 Data Availability and Privacy

Patient-level clinical data, original DICOM files, model weights, and institution-specific data-processing pipelines are not included in this repository because of data privacy and institutional restrictions.

The repository provides publicly releasable code components for model implementation, examination-level video aggregation, and Grad-CAM visualization.

Users should only apply the scripts to appropriately de-identified data and comply with applicable institutional and regulatory requirements.

## 📝 Citation

Citation information will be added after publication.

## 📧 Contact

For questions regarding the code, please contact:

Zhihan Yan: [yanzhihanwz@163.com](mailto:yanzhihanwz@163.com)

Dinggang Shen: [dgshen@shanghaitech.edu.cn](mailto:dgshen@shanghaitech.edu.cn)
