import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

class FashionMNISTClassifier:
    def __init__(self):
        # Class names for Fashion MNIST
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.model = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the Fashion MNIST dataset."""
        # Load the dataset
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        
        # Normalize pixel values to be between 0 and 1
        self.train_images = self.train_images.astype('float32') / 255.0
        self.test_images = self.test_images.astype('float32') / 255.0
        
        # Reshape images to include channel dimension
        self.train_images = self.train_images.reshape((-1, 28, 28, 1))
        self.test_images = self.test_images.reshape((-1, 28, 28, 1))
        
    def build_model(self):
        """Build the CNN model architecture."""
        self.model = models.Sequential([
            # Layer 1: Convolutional + MaxPooling
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Layer 2: Convolutional + MaxPooling
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Layer 3: Convolutional
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Layer 4: Flatten
            layers.Flatten(),
            
            # Layer 5: Dense
            layers.Dense(64, activation='relu'),
            
            # Layer 6: Output
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
    def train_model(self, epochs=10):
        """Train the model."""
        return self.model.fit(self.train_images, self.train_labels, 
                            epochs=epochs,
                            validation_data=(self.test_images, self.test_labels))
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print(f'\nTest accuracy: {test_acc}')
        return test_acc
    
    def predict_image(self, image_index):
        """Make prediction for a single image."""
        img = self.test_images[image_index]
        prediction = self.model.predict(img.reshape(1, 28, 28, 1))
        predicted_class = np.argmax(prediction)
        actual_class = self.test_labels[image_index]
        
        # Plot the image and predictions
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 1, 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {self.class_names[predicted_class]}\n' +
                 f'Actual: {self.class_names[actual_class]}')
        plt.axis('off')
        plt.show()
        
        return predicted_class, actual_class

def main():
    # Create classifier instance
    classifier = FashionMNISTClassifier()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    classifier.load_and_preprocess_data()
    
    # Build and train model
    print("\nBuilding model...")
    classifier.build_model()
    classifier.model.summary()
    
    print("\nTraining model...")
    history = classifier.train_model(epochs=10)
    
    # Evaluate model
    print("\nEvaluating model...")
    classifier.evaluate_model()
    
    # Make predictions
    print("\nMaking predictions...")
    for i in [0, 1]:  # Predict first two test images
        pred_class, actual_class = classifier.predict_image(i)
        print(f"\nImage {i+1}:")
        print(f"Predicted class: {classifier.class_names[pred_class]}")
        print(f"Actual class: {classifier.class_names[actual_class]}")

if __name__ == "__main__":
    main()
