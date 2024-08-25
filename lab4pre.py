import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración de directorios
base_dir = r'C:\Users\manue\OneDrive\Escritorio\Data_Science\Lab4\archive\PolyMNIST\MMNIST'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
modality_dirs = ['m0', 'm1', 'm2', 'm3', 'm4']

# Función para mostrar ejemplos de imágenes de cada modalidad
def mostrar_ejemplos_modalidad(modality_dirs, base_dir, num_images=5):
    fig, axes = plt.subplots(len(modality_dirs), num_images, figsize=(10, 10))
    for i, modality in enumerate(modality_dirs):
        modality_path = os.path.join(base_dir, modality)
        images = os.listdir(modality_path)[:num_images]
        for j, img_name in enumerate(images):
            img_path = os.path.join(modality_path, img_name)
            img = plt.imread(img_path)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
        axes[i, 0].set_title(modality)
    plt.show()

# Mostrar ejemplos del conjunto de entrenamiento
mostrar_ejemplos_modalidad(modality_dirs, train_dir)

# Mostrar ejemplos del conjunto de prueba
mostrar_ejemplos_modalidad(modality_dirs, test_dir)

# Análisis Exploratorio
def analisis_exploratorio(base_dir, modality_dirs):
    # Cálculo de la distribución de clases
    distribucion_clases = {modality: len(os.listdir(os.path.join(base_dir, modality))) for modality in modality_dirs}
    
    # Imprimir distribución de clases
    for modality, count in distribucion_clases.items():
        print(f"Modalidad {modality}: {count} imágenes")
    
    # Cálculo de la resolución de imágenes
    sample_image_path = os.path.join(base_dir, modality_dirs[0], os.listdir(os.path.join(base_dir, modality_dirs[0]))[0])
    sample_image = plt.imread(sample_image_path)
    print(f"Resolución de imagen de muestra: {sample_image.shape}")

# Realizar análisis exploratorio en el conjunto de entrenamiento
analisis_exploratorio(train_dir, modality_dirs)
