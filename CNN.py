# ============================================
# CNN IMAGE CLASSIFICATION PROJECT
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical


# ============================================
# 1. LOAD TRAINING DATASET
# ============================================

df = pd.read_csv("train.csv (1).zip")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Training Dataset:")
print(df.head())

print("\nTrain columns:")
print(df.columns.tolist())


# ============================================
# 2. LOAD TESTING DATASET
# ============================================

df_test = pd.read_csv("test.csv (1).zip")

# Remove extra spaces from column names
df_test.columns = df_test.columns.str.strip()

print("\nTesting Dataset:")
print(df_test.head())

print("\nTest columns:")
print(df_test.columns.tolist())


# ============================================
# 3. SEPARATE FEATURES AND LABEL
# ============================================

# Training data
X = df.drop("label", axis=1).values
y = df["label"].values

# Test data
# Test dataset does NOT contain label
X_test = df_test.values


# ============================================
# 4. CONVERT DATA INTO FLOAT
# ============================================

X = X.astype("float32")
X_test = X_test.astype("float32")


# ============================================
# 5. NORMALIZE PIXEL VALUES
# ============================================

X = X / 255.0
X_test = X_test / 255.0


# ============================================
# 6. RESHAPE DATA FOR CNN
# ============================================

# 784 pixels = 28 x 28 image
# 1 = grayscale channel

X_img = X.reshape(-1, 28, 28, 1)
X_test_img = X_test.reshape(-1, 28, 28, 1)


# ============================================
# 7. ONE-HOT ENCODE LABELS
# ============================================

y_cat = to_categorical(y, num_classes=10)


# ============================================
# 8. DISPLAY DATA SHAPES
# ============================================

print("\nData Shapes:")
print("X:", X.shape)
print("X_img:", X_img.shape)
print("y:", y.shape)
print("y_cat:", y_cat.shape)
print("X_test:", X_test.shape)
print("X_test_img:", X_test_img.shape)

print("\nData preprocessing successfully completed!")


perceptron = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(10, activation="softmax")
])


perceptron.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    
history_percp = perceptron.fit(
    X_img,
    y_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


acc_percp = perceptron.evaluate(X_img, y_cat, verbose=0)[1]
print(acc_percp)


#ANN
ann = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

acc_ann = ann.evaluate(X_img, y_cat, verbose=0)[1]
print(acc_ann)

X_train_cnn = X.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)


cnn = Sequential([
    Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history_cnn = cnn.fit(
    X_train_cnn,
    y_cat,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

acc_cnn = cnn.evaluate(X_train_cnn, y_cat, verbose=0)[1]
print(acc_cnn)


def show_side_by_side(models, names, X_test_img, X_test_cnn, n=5):

    plt.figure(figsize=(15, 8))

    for i in range(n):
        for j, model in enumerate(models):

            if j == 2:
                image = X_test_cnn[i]
            else:
                image = X_test_img[i]

            prediction = np.argmax(
                model.predict(np.expand_dims(image, axis=0), verbose=0),
                axis=1
            )[0]

            plt.subplot(n, 3, i * 3 + j + 1)
            plt.imshow(image.squeeze(), cmap="gray")
            plt.title(f"{names[j]}: {prediction}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()
show_side_by_side([perceptron, ann, cnn], ["Perceptron", "ANN", "CNN"], X_test_img, X_test_cnn, 5)