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
    st.write("**ข้อมูลชุดข้อมูล:**")
    st.write(
        """
        - **California Housing Dataset** เป็นชุดข้อมูลที่มีข้อมูลเกี่ยวกับราคาบ้านในรัฐแคลิฟอร์เนีย  
        - ประกอบด้วยคุณลักษณะหลายตัว เช่น จำนวนห้องพัก, อายุของบ้าน, ความหนาแน่นของประชากร ฯลฯ  
        - ชุดข้อมูลนี้ใช้ในการทำนายราคาบ้าน โดยมีตัวแปรตามเป็นราคาบ้าน (ในหน่วยพันดอลลาร์)   
        """
    )

    st.write("## แนวทางการพัฒนาโมเดล")
    st.write("### การเตรียมข้อมูล")
    st.write(
        """
        - ใช้ **California Housing Dataset จาก Scikit-learn** ซึ่งเป็นชุดข้อมูลเกี่ยวกับราคาบ้านในแคลิฟอร์เนีย
        - แยกคุณลักษณะ (X) และค่าตอบสนอง (y) โดย X ประกอบด้วยคุณลักษณะของบ้าน เช่น จำนวนห้อง, อายุของบ้าน, ความหนาแน่นของประชากร เป็นต้น ส่วน y คือราคาบ้าน
        - แบ่งข้อมูลเป็นชุดฝึก (80%) และชุดทดสอบ (20%) ด้วย `train_test_split()`
        - ใช้ **StandardScaler** ปรับขนาดข้อมูลให้เป็นสเกลมาตรฐาน **(Z-score Normalization)** เพื่อช่วยให้โมเดลมีประสิทธิภาพที่ดีขึ้น
        """)
    st.write("### ทฤษฎีของอัลกอริทึมที่พัฒนา")
    st.write("#### 1.Linear Regression")
    st.write(
        """
        เป็นโมเดลเชิงเส้นที่พยายามหาความสัมพันธ์ระหว่างตัวแปรอิสระ (X) กับตัวแปรตาม (y) 
        โดยอยู่ในรูปของสมการเส้นตรง:
        """
    )
    st.latex(r"y = w_1X_1 + w_2X_2 + ... + w_nX_n + b")
    st.write(
        """
        ใช้วิธี **Ordinary Least Squares (OLS)** เพื่อลดค่าความคลาดเคลื่อนรวม (Residual Sum of Squares)

        **ค่าที่วัดผลลัพธ์ของโมเดล:**
        - **Mean Squared Error (MSE):** คำนวณค่าความคลาดเคลื่อนของโมเดล
        - **R² Score:** วัดว่าโมเดลอธิบายความแปรปรวนของข้อมูลได้ดีเพียงใด
        """
    )
    st.write("#### 2.Decision Tree Regressor")
    st.write(
        """
        เป็นโมเดลที่ใช้ต้นไม้ตัดสินใจในการแบ่งข้อมูลเป็นช่วง ๆ โดยใช้เงื่อนไขที่ลด **Mean Squared Error (MSE)** มากที่สุด
        """
    )

    st.write("#### วิธีการตัดสินใจของต้นไม้:")
    st.write(
        """
        - คำนวณความคลาดเคลื่อนของข้อมูลแต่ละกลุ่ม  
        - เลือกคุณลักษณะที่แบ่งข้อมูลแล้วทำให้ค่าความคลาดเคลื่อนลดลงมากที่สุด  
        - ทำซ้ำไปเรื่อย ๆ จนถึงเงื่อนไขหยุด เช่น ขนาดกลุ่มข้อมูลที่เล็กเกินไป  

        **พารามิเตอร์ที่ใช้ในโค้ดนี้:**  
        - `min_samples_split=10` → กำหนดให้ต้องมีข้อมูลอย่างน้อย 10 ตัวก่อนที่โหนดจะถูกแบ่ง  
        - `min_samples_leaf=16` → กำหนดให้ใบของต้นไม้ต้องมีข้อมูลอย่างน้อย 16 ตัว เพื่อป้องกัน Overfitting  
        """
    )

    st.write("### ขั้นตอนการพัฒนา")
    st.write(
        """
        1. **Train โมเดล Linear Regression** ด้วย `model.fit(X_train, y_train)`  
        2. **Train โมเดล Decision Tree Regressor** ด้วย `dt_model.fit(X_train, y_train)`  
        3. **ทดสอบโมเดล** กับชุดข้อมูลทดสอบ `X_test`  
        4. **วัดผลลัพธ์ของโมเดล** ด้วยค่า **MSE** และ **R² Score**  
        5. **แสดงผลการทำนายของโมเดล** ผ่านกราฟ **Actual vs Predicted House Prices**  
           เพื่อดูว่าโมเดลทำนายได้ดีเพียงใด  
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
        EuroSAT Dataset มีคุณสมบัติเป็น **13 Spectral Bands** ที่ได้จากดาวเทียม Sentinel-2  
        แต่ละ Feature มีช่วงค่าความยาวคลื่นที่แตกต่างกัน ซึ่งช่วยให้สามารถจำแนกประเภทของภาพได้ดีขึ้น  
        """
    )
    st.write("**Feature:**")

    feature_data = {
        "Feature": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08a", "B09", "B10", "B11", "B12"],
        "รายละเอียด": [
            "Coastal aerosol", "Blue", "Green", "Red", "Vegetation red edge 1",
            "Vegetation red edge 2", "Vegetation red edge 3", "Near-infrared (NIR)",
            "Narrow NIR", "Water vapor", "Shortwave infrared (SWIR) - Cirrus",
            "SWIR 1", "SWIR 2"
        ],
        "ช่วงค่าของ Band (nm)": [
            "443", "490", "560", "665", "705", "740", "783", "842", "865", "945", "1375", "1610", "2190"
        ]
    }
    df_features = pd.DataFrame(feature_data)
    st.dataframe(df_features)

    st.write("**ข้อมูลชุดข้อมูล:**")
    st.write(
        """
        - **EuroSAT Dataset** เป็นชุดข้อมูลภาพถ่ายดาวเทียมจาก Sentinel-2  
        - แบ่งออกเป็น **10 หมวดหมู่** เช่น อาคาร, ป่าไม้, ทะเลสาบ ฯลฯ  
        - ใช้เป็น **Classification Task** สำหรับการจำแนกประเภทภาพ  
        """
    )

    st.write("## แนวทางการพัฒนาโมเดล")
    st.write("### **การเตรียมข้อมูล**")
    st.write(
        """
        - โหลดข้อมูลจาก TensorFlow Datasets (`tfds.load('eurosat', with_info=True, as_supervised=True)`)  
        - แบ่งข้อมูลเป็น **80% สำหรับ Train** และ **20% สำหรับ Test**  
        - ใช้ฟังก์ชัน `preprocess_image()` เพื่อ **Normalize** ค่า Pixel ให้อยู่ในช่วง **[0,1]**  
        - ใช้ **batch(32)** และ **prefetch(tf.data.AUTOTUNE)** เพื่อปรับปรุงประสิทธิภาพการประมวลผล  
        """
    )

    st.write("### ทฤษฎีของอัลกอริทึมที่พัฒนา")
    st.write("#### **Convolutional Neural Networks (CNNs)**")
    st.write(
        """
        - **CNNs** เป็นโมเดล Deep Learning ที่ใช้การ Convolution เพื่อลดขนาดภาพและดึงคุณลักษณะที่สำคัญออกมา  
        - แต่ละเลเยอร์ของ CNN ประกอบด้วย:  
            - **Convolutional Layer (Conv2D)** → สกัดคุณลักษณะจากภาพ  
            - **MaxPooling Layer (MaxPooling2D)** → ลดขนาดภาพเพื่อให้โมเดลเรียนรู้ได้เร็วขึ้น  
            - **Fully Connected Layer (Dense Layer)** → เชื่อมโยงคุณลักษณะเพื่อทำการจำแนกประเภท  
        """
    )


    st.write("### ขั้นตอนการพัฒนา")
    st.write(
        """
        1. **Train โมเดล CNN** ด้วย `cnn_model.fit(train_dataset, epochs=10, validation_data=test_dataset)`  
        2. **ทดสอบโมเดล** กับชุดข้อมูล `test_dataset`  
        3. **วัดผลลัพธ์ของโมเดล** ด้วยค่า **Test Accuracy** และ **Test Loss**  
        4. **แสดงผล Training vs Validation Accuracy** ด้วยกราฟ  
        5. **แสดงตัวอย่างภาพที่โมเดลทำนาย** พร้อมค่าจริงเพื่อดูผลลัพธ์  
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



    




