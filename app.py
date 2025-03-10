import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras import layers, models


# Sidebar for page selection
page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Train Models (Linear Regression & Decision Tree)", "Train Neural Network", "Documents"])

if page == "Data Overview":
    # California Housing Dataset (Scikit-learn)
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    st.write("### California Housing Dataset Overview")
    st.write("**Features:**")
    st.write(X.head())
    st.write("**Target (House Prices):**")
    st.write(pd.Series(y).head())

elif page == "Train Models (Linear Regression & Decision Tree)":
    # California Housing Dataset (Scikit-learn)
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)  # กำหนด X ให้ชัดเจน
    y = data.target  # กำหนด y ให้ชัดเจน

    # Split Data for Scikit-learn models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Linear Regression Model (Scikit-learn)
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)
    y_pred = model_sklearn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Decision Tree Regressor Model (Scikit-learn)
    dt_model = DecisionTreeRegressor(min_samples_split=10, min_samples_leaf=16, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)

    # Display Results
    st.write("### Linear Regression Model")
    st.write(f"**Mean Squared Error**: {mse:.4f}")
    st.write(f"**R² Score**: {r2:.4f}")

    # Plot the results for Linear Regression Model
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
    plt.title('Actual vs Predicted House Prices (Linear Regression)')
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    
    # Add text on the plot for Linear Regression MSE and R² Score
    plt.text(min(y_test), max(y_pred), f'MSE: {mse:.4f}\nR²: {r2:.4f}', fontsize=12, ha='left', va='top', color='blue')

    plt.legend()
    st.pyplot(plt)

    st.write("### Decision Tree Model")
    st.write(f"**Mean Squared Error**: {mse_dt:.4f}")
    st.write(f"**R² Score**: {r2_dt:.4f}")

    # Plot the results for Decision Tree Model
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred_dt, color='green', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
    plt.title('Actual vs Predicted House Prices (Decision Tree)')
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')

    # Add text on the plot for Decision Tree MSE and R² Score
    plt.text(min(y_test), max(y_pred_dt), f'MSE: {mse_dt:.4f}\nR²: {r2_dt:.4f}', fontsize=12, ha='left', va='top', color='green')

    plt.legend()
    st.pyplot(plt)



elif page == "Train Neural Network":
    # EuroSAT Dataset (TensorFlow Datasets)
    dataset, info = tfds.load('eurosat', with_info=True, as_supervised=True)
    
    # Data preparation for CNN
    train_size = int(0.8 * info.splits['train'].num_examples)
    train_dataset = dataset['train'].take(train_size)
    test_dataset = dataset['train'].skip(train_size)  # กำหนด test_dataset ให้ชัดเจน

    def preprocess_image(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
        return image, label

    train_dataset = train_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

    # CNN Model (TensorFlow)
    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # For classification
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = cnn_model.fit(train_dataset, epochs=10, validation_data=test_dataset)

    # Show training and validation accuracy over epochs
    st.write("### Model Training History")
    # Plotting training accuracy
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')

    st.pyplot(fig1)

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = cnn_model.evaluate(test_dataset)
    st.write(f"**Test Accuracy**: {test_accuracy:.4f}")
    st.write(f"**Test Loss**: {test_loss:.4f}")

    # Display a few sample images with their predicted labels
    st.write("### Sample Image Predictions")
    sample_image, sample_label = next(iter(test_dataset))
    predictions = cnn_model.predict(sample_image)

    # Plot a few images with both predicted and true labels
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(sample_image[i])
        true_label = sample_label[i].numpy()  # แสดงค่าจริง
        predicted_label = tf.argmax(predictions[i]).numpy()  # แสดงค่าทำนาย
        axes[i].set_title(f"True: {true_label}, Pred: {predicted_label}")
        axes[i].axis('off')

    st.pyplot(fig)



    st.write("### CNN Model Training Completed!")


elif page == "Documents":
    # Display documents or other information
    st.write("### Project Documentation")
    st.write("""
        This project involves training different machine learning models on two datasets:
        1. **California Housing Dataset** from Scikit-learn:
            - Used for Linear Regression and Decision Tree Regressor.
            - The task is to predict house prices based on various features.

        2. **EuroSAT Dataset** from TensorFlow Datasets:
            - Used for training a Convolutional Neural Network (CNN).
            - The task is to classify images into categories such as 'Forest', 'Highway', 'Residential', etc.
        
        The models are evaluated using appropriate metrics like Mean Squared Error (MSE), R² Score, and Accuracy.
        """)

