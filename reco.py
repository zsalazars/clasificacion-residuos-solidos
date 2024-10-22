import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Directorios de datos
train_dir = 'C:\\Users\\SERGIO\\Documents\\Python\\Proyectos\\residuos_solidos\\dataset\\train'
val_dir = 'C:\\Users\\SERGIO\\Documents\\Python\\Proyectos\\residuos_solidos\\dataset\\val'

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Preprocesamiento de imágenes
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Cargar el modelo preentrenado MobileNetV2 sin las capas superiores
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')

# Congelar las capas del modelo base para que no se actualicen durante el entrenamiento
base_model.trainable = False

# Añadir capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(4, activation='softmax')(x)  # 4 clases de residuos

model = Model(inputs=base_model.input, outputs=x)

# Compilar el modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Guardar el modelo entrenado
model.save('residuos_model.h5')
