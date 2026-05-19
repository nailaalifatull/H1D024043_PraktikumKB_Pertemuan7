#1. Library yang digunakan
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 2.Load dataset iris dari sklearn 
iris = load_iris()
X = iris.data    # 4 kolom pertama sebagai fitur
y = iris.target  # Kolom terakhir sebagai label

# 3. Mengonversi label dari string menjadi numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(iris.target_names[y])  # Mengubah label jadi 0, 1, 2

# 4. Memisahkan dataset menjadi data latih dan data validasi dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Buat model neural network 
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500,  activation='relu'),
    Dense(300,  activation='relu'),
    Dense(3,    activation='softmax')
])

model.summary()

# 6. Compile model 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Latih model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 8. Evaluasi model 
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# 9. Plot loss & accuracy 
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Training History')
plt.xlabel('Epoch')
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot training history disimpan ke training_history.png")

# 10. Prediksi 
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

print("Prediksi  :", predicted_classes)
print("Label Asli:", y_test)

# 11. Confusion Matrix 
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix disimpan ke confusion_matrix.png")

# 12. Prediksi data baru 
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width  = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width  = float(input("Masukkan petal width: "))

    # Membuat data array baru
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Melakukan prediksi
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)

    # Mengonversi hasil prediksi numerik menjadi label asli
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()