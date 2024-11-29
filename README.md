# Fashion MNIST Classification Project

This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset using Keras and TensorFlow.

## Project Structure

The project consists of a single Python file `fashion_mnist_classifier.py` that contains the complete implementation of the CNN classifier.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required packages using:

```bash
pip install tensorflow numpy matplotlib
```

## Implementation Details

The CNN architecture consists of six layers:
1. Convolutional Layer + MaxPooling
2. Convolutional Layer + MaxPooling
3. Convolutional Layer
4. Flatten Layer
5. Dense Layer
6. Output Layer (Dense)

The model uses:
- ReLU activation for hidden layers
- Softmax activation for the output layer
- Adam optimizer
- Sparse categorical crossentropy loss function

## Usage

1. Clone the repository:
```bash
git clone repository
cd fashion-mnist-classification
```

2. Run the main script:
```bash
python fashion_mnist_classifier.py
```

The script will:
- Load and preprocess the Fashion MNIST dataset
- Build and train the CNN model
- Evaluate the model's performance
- Make predictions on two test images
- Display the results with visualizations

## Output

The program will display:
- Model architecture summary
- Training progress (accuracy and loss)
- Final test accuracy
- Predictions for two test images with visualizations

## Customization

You can modify the following parameters in the code:
- Number of epochs (default: 10)
- Model architecture (layers, filters, etc.)
- Number of test predictions

## Model Performance

The model typically achieves:
- Training accuracy: ~90-93%
- Validation accuracy: ~88-91%

## Error Handling

The code includes basic error handling for:
- Dataset loading
- Model training
- Prediction making

