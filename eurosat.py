import tensorflow_datasets as tfds

# Try loading EuroSAT dataset
dataset, info = tfds.load('eurosat', with_info=True, as_supervised=True)
print(info)

import tensorflow as tf
from tensorflow.keras import layers, models

# Load EuroSAT dataset
dataset, info = tfds.load('eurosat', with_info=True, as_supervised=True)

# Split the data into training and testing (80% for training and 20% for testing)
train_size = int(0.8 * info.splits['train'].num_examples)
test_size = info.splits['train'].num_examples - train_size

# Split the data (80% for training, 20% for testing)
train_dataset = dataset['train'].take(train_size)
test_dataset = dataset['train'].skip(train_size)

# Data preprocessing (resize images and normalize the data)
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize images to range [0, 1]
    return image, label

# Prepare data by preprocessing images
train_dataset = train_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

# Build the CNN model
model = models.Sequential([
    # 1st Convolutional Layer: Convert image to feature maps
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    # 2nd Convolutional Layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3rd Convolutional Layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten: Flatten the data into a vector
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation='relu'),

    # Output Layer: 10 classes
    layers.Dense(10, activation='softmax')  # Using softmax for classification
])

# Summarize the architecture of the model
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset,
                    epochs=5,  # Number of epochs
                    validation_data=test_dataset)  # Use test dataset for validation during training

import matplotlib.pyplot as plt
# Make predictions for an image from the test dataset
sample_image, sample_label = next(iter(test_dataset))

# Select the first image from the batch
image = sample_image[9]
label = sample_label[9]

# Make a prediction with the model
prediction = model.predict(tf.expand_dims(image, axis=0))
predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

# Display the image
plt.imshow(image)
plt.title(f'True label: {label}, Predicted: {predicted_class}')
plt.axis('off')  # Hide axes
plt.show()

"""0.   AnnualCrop
1.   Forest
2.   HerbaceousVegetation  
3.   Highway  
4.   Industrial
5.   Pasture
6.   PermanentCrop
7.   Residential
8.   River
9.  SeaLake
"""

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Define class names
class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# Load EuroSAT RGB dataset
dataset, info = tfds.load('eurosat/rgb', with_info=True, as_supervised=True)

# Define train_dataset
train_dataset = dataset['train']

# Create a dictionary to store images from each class
class_images = {i: [] for i in range(10)}

# Fetch data from train_dataset and store images per class
for image, label in train_dataset:
    if len(class_images[label.numpy()]) < 10:  # Limit to 10 images per class
        class_images[label.numpy()].append(image)

    # Stop once 10 images are collected for each class
    if all(len(images) == 10 for images in class_images.values()):
        break

# Display images from each class
plt.figure(figsize=(15, 15))
for i, (class_id, images) in enumerate(class_images.items()):
    for j, image in enumerate(images):
        plt.subplot(10, 10, i * 10 + j + 1)
        plt.imshow(image.numpy())  # Display the image
        plt.title(class_names[class_id])  # Display class name
        plt.axis('off')

plt.tight_layout()
plt.show()
