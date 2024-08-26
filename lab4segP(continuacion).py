import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuración de directorios
base_dir = r'C:\Users\manue\OneDrive\Escritorio\Data_Science\Lab4\archive\PolyMNIST\MMNIST'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
modality_dirs = ['m0', 'm1', 'm2', 'm3', 'm4']

# Preparación del generador de datos con aumento
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Modelo CNN 1
model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 clases de salida
])

model1.compile(optimizer=Adam(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Entrenamiento del Modelo 1
print("Entrenando Modelo 1...")
history1_aug = model1.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
print("Modelo 1 terminado. Comenzando Modelo 2...")

# Modelo CNN 2
model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 clases de salida
])

model2.compile(optimizer=Adam(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Entrenamiento del Modelo 2
print("Entrenando Modelo 2...")
history2_aug = model2.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
print("Modelo 2 terminado.")

# Función para cargar los datos para el modelo K-NN
def cargar_datos_knn(base_dir, modality_dirs):
    X = []
    y = []
    for idx, modality in enumerate(modality_dirs):
        modality_path = os.path.join(base_dir, modality)
        images = os.listdir(modality_path)
        for img_name in images:
            img_path = os.path.join(modality_path, img_name)
            img = plt.imread(img_path).flatten()  # Convertir la imagen a un vector plano
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

# Carga de datos de entrenamiento y prueba para K-NN
print("Cargando datos para el modelo K-NN...")
X_train, y_train = cargar_datos_knn(train_dir, modality_dirs)
X_test, y_test = cargar_datos_knn(test_dir, modality_dirs)
print("Datos cargados. Iniciando entrenamiento del modelo K-NN...")

# Entrenamiento del modelo K-NN
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# Predicción y evaluación del modelo K-NN
y_pred = knn.predict(X_test)
print("Evaluación del modelo K-NN:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Comparación de Resultados
print("\nComparación de Modelos:")
print("Modelo 1 (CNN): Mejor precisión en validación: {:.2f}%".format(max(history1_aug.history['val_accuracy']) * 100))
print("Modelo 2 (CNN): Mejor precisión en validación: {:.2f}%".format(max(history2_aug.history['val_accuracy']) * 100))
print("Modelo K-NN: Precisión en prueba: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))