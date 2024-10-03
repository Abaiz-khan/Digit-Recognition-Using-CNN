# Digit Recoginition using CNN

This repository contains my project for classifying digits using a Convolutional Neural Network (CNN) trained on the Street View House Numbers (SVHN) dataset. The aim of this project is to develop a robust model that can accurately recognize digits in real-world scenarios, addressing challenges such as variations in lighting, scale, and rotation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Results and Evaluation](#results-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

### Model
The project utilizes a Convolutional Neural Network (CNN) architecture implemented using TensorFlow/Keras. The model is designed to learn complex patterns in the input images, enabling it to classify digits effectively.

### Dataset
The dataset used for this project is the Street View House Numbers (SVHN) dataset, which consists of over 600,000 labeled images of digits. The dataset is available for download [here](http://ufldl.stanford.edu/housenumbers/).

## Workflow

The workflow of the project includes the following stages:
1. **Data Acquisition**: Load the SVHN dataset.
2. **Data Preprocessing**: Normalize and augment the dataset.
3. **Model Building**: Define the CNN architecture.
4. **Model Training**: Train the model on the preprocessed dataset.
5. **Model Evaluation**: Assess the model's performance using various metrics.
6. **Conclusion and Future Work**: Summarize findings and suggest improvements.

## Results and Evaluation

The CNN model achieved a training accuracy of approximately 94.43% and a test accuracy of 94.34%. Various metrics such as precision, recall, and F1-score were computed to evaluate the model's performance. A confusion matrix was also generated to identify misclassifications.

## Installation

To run this project, you will need the following libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- SciPy
- Scikit-image

You can install these libraries using pip:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn scipy scikit-image
Usage
To use this project, clone the repository and run the provided Jupyter notebook or Python scripts:

bash
Copy code
git clone https://github.com/yourusername/SVHN-Digit-Classifier.git
cd SVHN-Digit-Classifier
Open the Jupyter notebook in your preferred environment and run the cells to train and evaluate the model.

Contributing
Contributions are welcome! If you have suggestions for improvements or enhancements, please create a pull request or open an issue.

css
Copy code

Feel free to modify any sections according to your preferences or project specifics!
