# Нейронные сети:
# - сверточные (конволюционные) нейронные сети (CNN) - компьютерное зрение, классификация изображений
# - рекуррентные нейронные сети (RNN) - распознование рукописного текста, обработка естественного языка
# - генеративные состязательные сети (GAN) - создание художественнх, музыкальных произведений
# - многослойный перцептрон - простейший тип нейронной сети

# НС работают только с действительными числами (c IR)

# 1. Начальные значения весов - случайные небольшие числа
# 2. Смещения = 0 (чаще всего)

# Для обучения нейронных сетей используют различные фреймворки:
# - TensorFlow (Keras)
# - PyTorch
# на выходе имеем модели

# Для прогнозирования:
# - TF Lite -> моб. устр.
# - TF VS

# План работы
# 1. Организация данных (обучающая, проверочная + контрольная группы)
# 2. Построим пайплайн подготовки
# 3. Аугментация данных - обогащение набора
# 4. Определение модели. Заморозка коэффициентов. Алгоритм оптимизатора, метрика оценки.
# 5. Обучение модели -> итерации -> пока метрика не станет приемлемой
# 6. Сохранение модели



import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.preprocessing import image

TRAIN_DATA_DIR = './train_data/'
VALIDATION_DATA_DIR = './val_data/'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HIGHT = 224, 224
BATCH_SIZE = 64

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

val_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=12345,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)

from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

def model_maker():
    base_model=MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable=False

    input = Input(shape=(IMG_WIDTH, IMG_HIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    prediction = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=prediction)


from tensorflow.keras.optimizers import Adam


model = model_maker()
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

import math

num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=num_steps,
    epochs=10,  # шаг обучнеия
    validation_data=val_generator,
    validation_steps=num_steps
)

print(val_generator.class_indices)
model.save('./model.h5')




