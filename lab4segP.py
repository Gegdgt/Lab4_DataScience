import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Reentrenamiento del Modelo 1 con Aumento de Datos
history1_aug = model1.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

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

# Reentrenamiento del Modelo 2 con Aumento de Datos
history2_aug = model2.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)