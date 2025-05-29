import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Генерация обучающих данных
# Создаю массив их 10000 пар чисел в диапазоне от 0 до 10 включительно
X = np.random.randint(0, 11, size=(10000, 2))   # 10000 пар чисел (x1, x2)

# Создаю массив с метками - суммами пар чисел
y = np.sum(X, axis=1)                           # y = x1 + x2

# Проведу нормализацию для повышения качества обучения
# Просто приведу числа к диапазону от 0 до 1, на меньших диапазонах качество обучения лучше
X = X / 10.0  # максимум слагаемых - 10
y = y / 20.0  # максимум суммы — 20

# Создание модели
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')  # Предсказание суммы
])

# Компиляция
model.compile(optimizer=Adam(),
              loss='mse',
              metrics=['mae'])

# Обучение
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)

# Сохраним модель
model.save('./sum_of_2_model.keras')

# Пример для теста
x_test = np.array([[3, 7], [5, 2]]) / 10.0
predictions = model.predict(x_test) * 20.0  # Обратно из нормализованного значения
print(predictions)


