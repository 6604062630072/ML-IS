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
page = st.sidebar.selectbox("Choose a page", ["Documents 1", "Models (Linear Regression & Decision Tree)", "Documents 2", "Models Neural Network(CNNs)"])

if page == "Documents 1":
    # California Housing Dataset (Scikit-learn)
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    st.write("# Models (Linear Regression & Decision Tree)")
    st.write("### **Dataset: Scikit-learn(California Housing Dataset)**")
    st.write("**Features:**")
    st.write(X.head())
    st.write("**Target (House Prices):**")
    st.write(pd.Series(y).head())
    st.write("**Dataset Description:**")
    st.write(
        """
        - **California Housing Dataset** is a dataset that contains information about house prices in California  
        - It includes several features like the number of rooms, age of the house, population density, etc.  
        - The dataset is used to predict house prices, where the target variable is the house price (in thousands of dollars).   
        """
    )

    st.write("## Model Development Approach")
    st.write("### Data Preparation")
    st.write(
        """
        - We use the **California Housing Dataset from Scikit-learn**, which contains information about housing prices in California
        - The features (X) include characteristics of the house such as number of rooms, age of the house, population density, etc., while the target variable (y) is the house price
        - We split the data into training (80%) and testing (20%) sets using `train_test_split()`
        - We use **StandardScaler** to normalize the data using **Z-score Normalization** to improve the model's performance
        """
    )
    st.write("### Algorithm Theory for Developed Models")
    st.write("#### 1. Linear Regression")
    st.write(
        """
        It is a linear model that tries to find the relationship between independent variables (X) and dependent variable (y) 
        in the form of a linear equation:
        """
    )
    st.latex(r"y = w_1X_1 + w_2X_2 + ... + w_nX_n + b")
    st.write(
        """
        **Model Evaluation Metrics:**
        - **Mean Squared Error (MSE):** Measures the deviation of the model
        - **R² Score:** Measures how well the model explains the variance of the data
        """
    )
    st.write("#### 2. Decision Tree Regressor")
    st.write(
        """
        It is a model that uses a decision tree to split data into intervals, using conditions that minimize **Mean Squared Error (MSE)**
        """
    )

    st.write("#### Decision Tree Split Process:")
    st.write(
        """
        - It calculates the error for each group of data  
        - Selects the feature that minimizes the error after splitting  
        - Continues splitting until stopping conditions are met (e.g., too few samples in a group)  

        **Parameters used in this code:**  
        - `min_samples_split=10` → Requires at least 10 samples before a node is split  
        - `min_samples_leaf=16` → Ensures leaf nodes have at least 16 samples to prevent overfitting  
        """
    )

    st.write("### Development Steps")
    st.write(
        """
        1. **Train the Linear Regression model** using `model.fit(X_train, y_train)`  
        2. **Train the Decision Tree Regressor model** using `dt_model.fit(X_train, y_train)`  
        3. **Test the models** on the test dataset `X_test`  
        4. **Evaluate the models** using **MSE** and **R² Score**  
        5. **Display the model's prediction results** through a graph of **Actual vs Predicted House Prices**  
           to evaluate the model's performance  
        """
    )



elif page == "Models (Linear Regression & Decision Tree)":
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



elif page == "Documents 2":
    # Display documents or other information
    st.write("# Models Neural Network (CNNs)")
    st.write("### **Dataset: EuroSAT (TensorFlow Datasets)**")
    st.write(
        """
        The EuroSAT Dataset contains **13 Spectral Bands** obtained from the Sentinel-2 satellite  
        Each feature has a different wavelength range, which helps in better image classification  
        """
    )
    st.write("**Features:**")

    feature_data = {
        "Feature": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08a", "B09", "B10", "B11", "B12"],
        "Description": [
            "Coastal aerosol", "Blue", "Green", "Red", "Vegetation red edge 1",
            "Vegetation red edge 2", "Vegetation red edge 3", "Near-infrared (NIR)",
            "Narrow NIR", "Water vapor", "Shortwave infrared (SWIR) - Cirrus",
            "SWIR 1", "SWIR 2"
        ],
        "Band Wavelength (nm)": [
            "443", "490", "560", "665", "705", "740", "783", "842", "865", "945", "1375", "1610", "2190"
        ]
    }
    df_features = pd.DataFrame(feature_data)
    st.dataframe(df_features)

    st.write("**Dataset Information:**")
    st.write(
        """
        - **EuroSAT Dataset** consists of satellite imagery from Sentinel-2  
        - It is classified into **10 categories** such as buildings, forests, lakes, etc.  
        - This dataset is used for **Classification Task** for image classification  
        """
    )

    st.write("## Model Development Approach")
    st.write("### **Data Preparation**")
    st.write(
        """
        - Load data from TensorFlow Datasets (`tfds.load('eurosat', with_info=True, as_supervised=True)`)  
        - Split data into **80% for Training** and **20% for Testing**  
        - Use the function `preprocess_image()` to **Normalize** pixel values in the range of **[0,1]**  
        - Use **batch(32)** and **prefetch(tf.data.AUTOTUNE)** to optimize processing speed  
        """
    )

    st.write("### Algorithm Theory for Developed Models")
    st.write("#### **Convolutional Neural Networks (CNNs)**")
    st.write(
        """
        - **CNNs** are deep learning models that use convolution operations to reduce image size and extract important features  
        - Each layer of a CNN includes:  
            - **Convolutional Layer (Conv2D)** → Extracts features from the image  
            - **MaxPooling Layer (MaxPooling2D)** → Reduces image size to speed up learning  
            - **Fully Connected Layer (Dense Layer)** → Links features to perform classification  
        """
    )


    st.write("### Development Steps")
    st.write(
        """
        1. **Train the CNN model** with `cnn_model.fit(train_dataset, epochs=10, validation_data=test_dataset)`  
        2. **Test the model** on the `test_dataset`  
        3. **Evaluate the model's performance** using **Test Accuracy** and **Test Loss**  
        4. **Display training vs validation accuracy** through a graph  
        5. **Show sample images with model predictions** and compare with actual values  
        """
    )




elif page == "Models Neural Network(CNNs)":
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
    history = cnn_model.fit(train_dataset, epochs=8, validation_data=test_dataset)

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

    st.write(
        """
            0.   AnnualCrop
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
    )



    




