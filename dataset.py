# -*- coding: utf-8 -*-
"""ASL Gesture Recognition - Complete Training and Inference System"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

# Set paths
path = 'C:/Users/Mohit Pardeshi/OneDrive/Desktop/project'
train_dir = os.path.join(path, 'ASL_Dataset/Train')
test_dir = os.path.join(path, 'ASL_Dataset/Test')

print("=== ASL Gesture Recognition System ===")
print(f"Training data path: {train_dir}")
print(f"Test data path: {test_dir}")

# Get class names and validate dataset
def validate_dataset():
    """Validate and display dataset information"""
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"\nFound {len(class_names)} classes: {class_names}")
    
    # Count images in each class
    print("\nDataset Statistics:")
    print("-" * 50)
    total_train = 0
    total_test = 0
    
    for class_name in class_names:
        train_class_path = os.path.join(train_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)
        
        train_count = len([f for f in os.listdir(train_class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        test_count = len([f for f in os.listdir(test_class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        total_train += train_count
        total_test += test_count
        
        print(f"{class_name:>8}: {train_count:>5} train, {test_count:>4} test")
    
    print("-" * 50)
    print(f"Total: {total_train:>5} train, {total_test:>4} test images")
    
    return class_names

# Validate dataset
class_names = validate_dataset()

# Enhanced data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Don't flip ASL signs
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Simple rescaling for validation
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load data generators
print("\nLoading data generators...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Increased image size for better features
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {valid_generator.samples}")

# Create improved model architecture
def create_model(num_classes):
    """Create an improved CNN model for ASL recognition"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile model
print("\nCreating model...")
model = create_model(len(class_names))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Training callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

# Train the model
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=valid_generator,
    callbacks=callbacks,
    verbose=1
)

# Save the trained model
model_save_path = os.path.join(path, 'asl_model.h5')
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

# Plot training history
def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'training_history.png'))
    plt.show()

# Plot training history
plot_training_history(history)

# Evaluate model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

evaluation = model.evaluate(valid_generator, verbose=1)
print(f"Test Accuracy: {evaluation[1]:.4f}")
print(f"Test Loss: {evaluation[0]:.4f}")

# Real-time inference
print("\n" + "="*50)
print("STARTING REAL-TIME INFERENCE")
print("="*50)
print("Press 'q' to quit the webcam")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Create a copy for display
    display_frame = frame.copy()
    
    # Preprocess frame for prediction
    image = cv2.resize(frame, (128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    # Make prediction
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Get top 3 predictions
    top_3_idx = np.argsort(prediction[0])[-3:][::-1]
    
    # Display results
    cv2.putText(display_frame, f'Prediction: {class_names[predicted_class]}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, f'Confidence: {confidence:.1f}%', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show top 3 predictions
    y_offset = 90
    for i, idx in enumerate(top_3_idx):
        if i > 0:  # Skip the first one as it's already shown above
            class_name = class_names[idx]
            conf = prediction[0][idx] * 100
            cv2.putText(display_frame, f'{class_name}: {conf:.1f}%', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
    
    # Add instructions
    cv2.putText(display_frame, "Press 'q' to quit", 
                (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('ASL Gesture Recognition', display_frame)
    
    # Break on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nInference completed!")