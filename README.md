# Breast Cancer Classification from Ultrasound Images

![Breast Ultrasound Sample](https://via.placeholder.com/400x200?text=Breast+Ultrasound+Images)  
*Example ultrasound images from the dataset*

This project implements a deep learning solution for classifying breast ultrasound images into three categories: benign, malignant, and normal. The model achieves **98.65% accuracy** on the validation set using a custom CNN architecture.

## Key Features

- 🩺 **Medical Image Classification**: Identifies breast abnormalities in ultrasound images
- ⚙️ **Advanced Preprocessing**: Includes histogram equalization, bilateral filtering, and K-means segmentation
- ⚖️ **Class Balancing**: Uses data augmentation to address class imbalance
- 🧠 **Custom CNN Model**: Simple yet effective architecture for high accuracy
- 📊 **Performance Metrics**: 98.65% validation accuracy with minimal loss

## Dataset

The project uses the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) from Kaggle, which contains:

- 437 benign images
- 210 malignant images
- 133 normal images

## Preprocessing Pipeline

1. **Resizing**: All images resized to 224×224 pixels
2. **Histogram Equalization**: Enhances image contrast
3. **Gaussian Blur**: Reduces noise while preserving edges
4. **Bilateral Filtering**: Edge-preserving smoothing
5. **K-means Segmentation**: 2-cluster segmentation to highlight regions of interest

## Data Augmentation

To address class imbalance, we generate additional samples using:
- Rotation (±15°)
- Zoom (10% range)
- Width/height shifting (10% range)
- Horizontal flipping

After augmentation, each class contains approximately 850-900 images.

## Model Architecture

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])
```

## Performance

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------------|----------------|
| 1     | 62.68%            | 98.65%              | 0.0585         |
| 2     | 88.44%            | 99.42%              | 0.0268         |
| 5     | 99.71%            | 99.42%              | 0.0291         |
| 10    | 99.96%            | 98.65%              | 0.0516         |

**Final Validation Accuracy: 98.65%**

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- scikit-learn
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MariaSy19/Breast-Cancer-Classifier-.git
cd breast-cancer-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and place it in the `data/` directory

## Usage

1. Run the preprocessing and training script:
```bash
python breast_cancer_classification.py
```

2. To use the trained model for prediction:
```python
from model_utils import predict_breast_cancer

prediction = predict_breast_cancer('path/to/ultrasound_image.png')
print(f"Predicted class: {prediction}")
```

## File Structure

```
breast-cancer-classification/
├── data/                   # Dataset directory
├── outputs/                # Training outputs and models
├── utils/
│   ├── preprocessing.py    # Image processing functions
│   ├── augmentation.py     # Data augmentation functions
│   └── model_utils.py      # Model loading and prediction
├── breast_cancer_classification.ipynb  # Main Jupyter notebook
├── breast_cancer_classification.py     # Python script version
├── requirements.txt        # Dependencies
└── README.md               # This file
```


## Acknowledgments

- Dataset provided by [Dataset_BUSI_with_GT](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- Research paper: [Breast Ultrasound Images Dataset](https://www.sciencedirect.com/science/article/pii/S2352340919312181)