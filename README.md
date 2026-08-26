PH-PAS-Echocardiography
PH-PAS is a guideline-based pulmonary hypertension (PH) probability assessment framework for multiview transthoracic echocardiography (TTE) videos. The public code in this repository implements the core training components used to learn low-, intermediate-, and high-probability PH representations from routine TTE videos.
Overview
The released implementation includes:
A 3D ResNet-18 video encoder initialized with Kinetics-400 pretrained weights
Echocardiographic view embeddings
Imaging-modality embeddings
Masked-report-assisted video-text contrastive learning
A three-class classifier for low, intermediate, and high echocardiographic PH probability
Structured clinical variables are not used in the released PH-PAS model.
Model inputs
Training
During training, the model uses:
TTE videos
Echocardiographic view information
Imaging-modality information
De-identified and masked echocardiographic reports for auxiliary video-text contrastive learning
The text branch is used only to improve representation learning during training.
Inference
During inference, PH-PAS uses only:
TTE videos
Echocardiographic view information
Imaging-modality information
Echocardiographic reports and structured clinical variables are not required for inference.
Echocardiographic inputs
The study framework uses standard multiview TTE videos, including:
Parasternal long-axis (PLAX)
Parasternal short-axis (PSAX)
Apical four-chamber (A4C)
Supported imaging modes include:
B-mode
Colour Doppler flow imaging (CDFI)
Each eligible video is uniformly sampled to 20 frames and resized to 224 × 224 pixels before model input.
PH probability classes
The classifier outputs three echocardiographic PH probability categories:
`0` = Low
`1` = Intermediate
`2` = High
These categories correspond to guideline-based echocardiographic PH probability stratification.
Masked-report-assisted contrastive learning
Echocardiographic reports are used only during training as an auxiliary source of semantic supervision.
Before text encoding, reports should be:
De-identified
Masked to remove direct PH diagnosis statements
Masked to remove explicit PH probability labels
Masked to remove TRV values and estimated pulmonary artery pressures
Masked to remove other information that could directly reveal the target label
Masked reports are encoded using `Chinese-BERT-wwm-ext`. Video and text representations are projected into a shared 512-dimensional embedding space and optimized using a symmetric video-to-text and text-to-video contrastive objective.
The training objective is:
```text
L\_total = L\_con + 3.74 × L\_cls
```
where:
`L\_con` is the symmetric video-text contrastive loss
`L\_cls` is the three-class cross-entropy classification loss
The contrastive temperature is set to `0.07`.
Public code scope
This repository intentionally provides the core model-training implementation only.
Included:
Video encoder
View embedding
Modality embedding
Text encoder for masked-report contrastive learning
Three-class PH probability classifier
Training loop and optimization settings
Not included:
Patient data
Original TTE videos
Private preprocessing pipelines
Model weights
RHC validation data
Reader-study data
Repository structure
```text
PH-PAS-Echocardiography/
├── README.md
├── ph\_pas\_public\_train.py
├── requirements.txt
└── .gitignore
```
Expected CSV format
The public training script expects training and development CSV files derived from the model-development cohort.
Required columns:
Column	Description
`VideoPath`	Path to the echocardiographic video
`PH\_label`	`0=Low`, `1=Intermediate`, `2=High`
`View`	Echocardiographic view label
`Modality`	Imaging-modality label
`masked\_report`	De-identified and masked echocardiographic report
Internal-test and external-validation datasets should not be used for model training, model selection, or hyperparameter optimization.
Installation
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Training example
```bash
python ph\_pas\_public\_train.py \\
  --train\_csv train.csv \\
  --dev\_csv dev.csv \\
  --output\_dir checkpoints
```
Optional arguments include:
```text
--text\_model
--epochs
--batch\_size
--num\_workers
--seed
```
The default text encoder is:
```text
hfl/chinese-bert-wwm-ext
```
Main training settings
The public script follows the principal training settings used in the study:
Video backbone: 3D ResNet-18
Pretraining: Kinetics-400
Frames per video: 20
Input size: 224 × 224
Projection dimension: 512
Batch size: 16
Optimizer: AdamW
Initial learning rate: `1e-4`
Weight decay: `1e-2`
Contrastive temperature: `0.07`
Classification-loss weight: `3.74`
Learning-rate scheduler: ReduceLROnPlateau
Scheduler factor: `0.5`
Scheduler patience: `3`
Data availability
The clinical and echocardiographic datasets used in the study are not publicly distributed in this repository because they contain protected clinical and imaging information and are subject to institutional and ethical restrictions.
Code availability
This repository provides a simplified research implementation of the core PH-PAS training framework described in the associated manuscript. Certain study-specific data-processing, patient-level aggregation, validation, and analysis components are not included.
Citation
Citation information will be added after publication.
Disclaimer
This code is provided for research purposes only. It is not intended for direct clinical diagnosis, treatment decisions, or independent determination of right heart catheterization indications.
