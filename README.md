# ML-lab-prject
#Objective:
Classify images as either dogs or cats using a pre-trained CNN model (e.g., VGG16, ResNet50, or MobileNetV2) with transfer learning.

#Dataset:
We'll use a mini Dog vs Cat dataset (~2 MB, 2000 images: 1000 cats, 1000 dogs).

Download Link (small-sized):
https://www.microsoft.com/en-us/download/details.aspx?id=54765
Use the "Kaggle Cats and Dogs Dataset" → Filter a small portion (e.g., 1000 cat + 1000 dog images).

Or use directly from Kaggle (resize/filter in code):
https://www.kaggle.com/datasets/c/dogs-vs-cats

Model: Transfer Learning with VGG16

pip install tensorflow keras matplotlib
Python Code (Simplified End-to-End):
python
Copy
Edit
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import os

# Set up paths
base_dir = '/path_to_dataset'  # replace with your dataset path
train_dir = os.path.join(base_dir, 'train')  # e.g., train/cats, train/dogs

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training')

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Load pre-trained VGG16 model without top
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False  # freeze base

# Add custom layers on top
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator)

# Evaluate
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc*100:.2f}%")
📊 Expected Output:
Accuracy: ~85–95% (depending on size and epochs)

Model: Fast to train due to frozen base

Perfect for Lab: Small size, clean structure, demonstrates transfer learning

