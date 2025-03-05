# Sea Animals Classifier

## Overview

This repository contains a Jupyter Notebook implementing a machine learning model to classify sea animals based on given data. The project explores different machine learning techniques and evaluates their performance.

## Features

- Data preprocessing and exploration
- Model training and evaluation
- Use of popular ML libraries such as TensorFlow, Scikit-learn, and Pandas
- Performance metrics and visualization

## Technology & Techniques

- **Programming Languages:** Python
- **Libraries Used:** TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning Techniques:**
  - Supervised learning approach with deep learning models
  - CNN (Convolutional Neural Networks) for image classification
  - Data augmentation for better generalization
  - Hyperparameter tuning using Bayesian Optimization Tuner
  - Model evaluation with weighted precision
  - Transfer learning with pre-trained models such as ResNet, DenseNet and EfficientNet

## Development Pipeline

1. **Data Collection & Preprocessing:**

   - Gather image datasets of sea animals
   - Perform data cleaning and augmentation
   - Split dataset into training, validation, and test sets

2. **Exploratory Data Analysis (EDA):**

   - Visualize class distribution
   - Identify potential data imbalances and take corrective measures

3. **Model Selection & Training:**

   - Implement baseline models for performance comparison
   - Train CNN-based deep learning models
   - Apply transfer learning with pre-trained architectures
   - Optimize hyperparameters for best performance

4. **Evaluation & Validation:**

   - Use weighted precision for performance assessment
   - Conduct cross-validation to ensure robustness

5. **Model Deployment:**

   - Export the trained model
   - Integrate with an API or application for real-world use
   - Deploy on cloud platforms if required

## Installation

To run this project locally, ensure you have Python installed and then follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/zythedeveloper/sea-animals-classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sea-animals-classifier
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `sea_animals_classifier.ipynb` and run the cells.

## Screenshots
### Simple Flask Application
<img src="https://drive.google.com/uc?id=1lQVWVbFwBVwy_JRipfxeoFLfV6yXLVuE" width="600px" />
<img src="https://drive.google.com/uc?id=1e3fawVOlc5XFnAwg-yiz-yLEB2di3DYi" width="600px" />

## Model Performance Comparison

| Model             | Weighted Prec.  | Top-1 Prec.      | Top-5 Prec.      | Loss         |
|------------------|---------------|----------------|----------------|-------------|
| DenseNet-201     | 84.73%        | 90.60%         | 96.94%         | 0.55563     |
| **DenseNet-201** | **89.79%** (↑ 5.06%) | **93.13%** (↑ 2.53%) | **98.03%** (↑ 1.09%) | **0.38890** (↓ 0.16673) |
| EfficientNet-B3  | 83.80%        | 91.77%         | 96.21%         | 0.61064     |
| **EfficientNet-B3** | **86.50%** (↑ 2.70%) | **92.61%** (↑ 0.84%) | **97.30%** (↑ 1.09%) | **0.49626** (↓ 0.11438) |
| ResNet-152       | 78.68%        | 88.01%         | 94.39%         | 0.79901     |
| **ResNet-152**   | **78.43%** (↓ 0.25%) | **85.45%** (↓ 2.56%) | **94.39%** | **0.80210** (↑ 0.00309) |

Note that: Normal means manually tuned whereas Underline means fine-tuned models.

## Usage

- Modify the dataset or features in the preprocessing section as needed.
- Train the model and analyze performance.
- Export and use the trained model for classification.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
