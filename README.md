# Comparative Analysis of Transfer Learning Models for Breast Cancer Classification
This repository contains the codebase for our paper, **"Comparative Analysis of Transfer Learning Models for Breast Cancer Classification"**. Our study explores and evaluates various deep learning architectures for accurately classifying histopathological images, focusing on distinguishing **Invasive Ductal Carcinoma (IDC)** from non-IDC, with the goal of enhancing breast cancer diagnosis.
## Project Overview

Histopathology image classification plays a critical role in the early detection and precise diagnosis of breast cancer. In this work, we rigorously compare the performance of eight prominent transfer learning models—**ResNet-50, DenseNet-121, ResNeXt-50, Vision Transformer (ViT), GoogLeNet (Inception v3), EfficientNet, MobileNet, and SqueezeNet**—on a dataset of 277,524 image patches. Each model is evaluated for:
- **Accuracy**
- **Computational Efficiency**
- **Robustness**

Key findings highlight the superior accuracy of attention-based models, particularly the **Vision Transformer (ViT)**, which achieved a validation accuracy of **93%**, outperforming traditional convolutional networks. This repository provides scripts and setup instructions to reproduce our results, demonstrating the practical application of advanced machine learning techniques in clinical settings.
## Repository Structure

```plaintext
Comparative-Analysis-Transfer-Learning-Breast-Cancer
├── data/                 # Placeholder or instructions for dataset preparation
├── models/               # Model architecture definitions and (optional) pre-trained weights
├── notebooks/            # Jupyter notebooks for exploratory data analysis (EDA) and visualization
├── src/                  # Core scripts for training, validation, and evaluation
├── requirements.txt      # Required dependencies
├── README.md             # Detailed overview of the project
├── results/              # Logs, graphs, and validation metrics
└── LICENSE               # License information for the repository
## Installation and Setup

### Prerequisites
Ensure you have the following libraries installed:
- Python 3.8+
- PyTorch
- TensorFlow
- Hugging Face's Transformers library (for Vision Transformer)
- OpenCV (for data preprocessing)

### Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
## Data Preparation

1. **Dataset**: Download the dataset (277,524 histopathology image patches). Detailed instructions or a download link can be found [here](link to dataset).
2. **Preprocessing**: Resize, normalize, and extract patches as needed. Ensure images are organized in the `data/` directory, structured as per the `data` folder's requirements.
## Model Training and Evaluation

Each model can be trained and evaluated using the following command:
```bash
python src/train.py --model <model_name> --epochs <num_epochs>
python src/train.py --model resnet50 --epochs 30
## Results and Key Findings

| Model       | Validation Accuracy | Inference Time | Params (Millions) |
|-------------|---------------------|----------------|--------------------|
| ResNet-50   | 89%                 | Fast           | 25.6              |
| ViT         | 93%                 | Moderate       | 85.8              |
| DenseNet-121| 90%                 | Moderate       | 8.1               |
| ...         | ...                 | ...            | ...               |

- **Vision Transformer (ViT)** achieved the highest validation accuracy, demonstrating the advantages of attention-based architectures over traditional convolutional networks in this classification task.
## Model Interpretability and Practical Application

Understanding model decisions is crucial in clinical applications. Techniques like **Grad-CAM** or **SHAP** can be applied to interpret the model’s decision-making process, providing insights into which regions of the images contribute most significantly to the diagnosis. Refer to `notebooks/interpretable_visualizations.ipynb` for examples.
## Contributing

Contributions to this project are welcome. If you'd like to add new models, improve efficiency, or help with documentation, please fork the repository and submit a pull request.
## Citation

If you use our code, please cite our paper:

```plaintext
@article{your_paper,
  title={Comparative Analysis of Transfer Learning Models for Breast Cancer Classification},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:2408.16859},
  year={2024}
}











