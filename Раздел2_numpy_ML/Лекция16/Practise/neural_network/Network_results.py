import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import load_model

model = load_model('./sum_of_2_model.keras')

# Ввод чисел от пользователя
try:
    x1 = float(input("Введите первое число (от 0 до 10): "))
    x2 = float(input("Введите второе число (от 0 до 10): "))

    if not (0 <= x1 <= 10 and 0 <= x2 <= 10):
        raise ValueError("Числа должны быть от 0 до 10.")

    # Подготовка данных: нормализация
    x_input = np.array([[x1, x2]]) / 10.0

    # Предсказание
    y_pred = model.predict(x_input)
    predicted_sum = y_pred[0][0] * 20.0  # денормализация

    # Вывод результата
    print(f"Предсказанная сумма: {predicted_sum:.2f}")

except ValueError as e:
    print(f"Ошибка ввода: {e}")