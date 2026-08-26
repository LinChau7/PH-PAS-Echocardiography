# PH-PAS-Echocardiography

## AI-driven Pulmonary Hypertension Probability Assessment using Multiview Echocardiography

This repository contains the PyTorch implementation of PH-PAS, an AI framework for guideline-based pulmonary hypertension (PH) probability assessment from multiview transthoracic echocardiography (TTE) videos.

## 📋 Overview

PH-PAS integrates:

* **Echocardiographic Videos:** Spatio-temporal feature extraction using 3D ResNet-18.
* **View and Modality Embeddings:** Echocardiographic view and imaging modality information are incorporated into the video representation.
* **Vision-Language Pretraining:** Masked echocardiographic reports are used during training for video-text contrastive learning.

During inference, the model requires only TTE videos together with view and modality information.

## ⚙️ Project Structure

```text

├── ph\_pas\_public\_train.py

├── requirements.txt

├── .gitignore

└── README.md

```

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

| Column          | Description                                       |
| --------------- | ------------------------------------------------- |
| `VideoPath`     | Path to the echocardiographic video               |
| `PH\_label`      | 0 = Low, 1 = Intermediate, 2 = High               |
| `View`          | Echocardiographic view                            |
| `Modality`      | Imaging modality                                  |
| `masked\_report` | De-identified and masked echocardiographic report |

The model uses standard multiview TTE videos, including:

* Parasternal long-axis (PLAX)
* Parasternal short-axis (PSAX)
* Apical four-chamber (A4C)

Imaging modalities include:

* B-mode
* Colour Doppler flow imaging (CDFI)

Each video is uniformly sampled to 20 frames and resized to 224 × 224 pixels.

## 🚀 Usage

### Training

```bash

python ph\_pas\_public\_train.py \\

&#x20; --train\_csv train.csv \\

&#x20; --dev\_csv dev.csv \\

&#x20; --output\_dir checkpoints

```

Optional arguments include:

```text

\--text\_model

\--epochs

\--batch\_size

\--num\_workers

\--seed

```

The default text encoder is:

```text

hfl/chinese-bert-wwm-ext

```

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

L\_total = L\_con + 3.74 × L\_cls

```

where `L\_con` denotes the symmetric video-text contrastive loss and `L\_cls` denotes the three-class cross-entropy classification loss.

The contrastive temperature is set to `0.07`.

### Inference

During inference, the text encoder is removed.

PH-PAS uses only:

* TTE videos
* View information
* Modality information

## 📝 Citation

Citation information will be added after publication.

## 📧 Contact

For questions regarding the code, please contact:

Zhihan Yan: [yanzhihanwz@163.com](mailto:yanzhihanwz@163.com)

Dinggang Shen: [dgshen@shanghaitech.edu.cn](mailto:dgshen@shanghaitech.edu.cn)
