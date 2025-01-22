import numpy as np

## Q1: Что надо изменить в последнем примере, чтобы он заработал без ошибок?
a = np.ones((3, 2))
b = np.arange(3)

print(a, a.ndim, a.shape)
print(b, b.ndim, b.shape)
# 2 (3, 2) -> (3, 2)    (3, 2)
# 1 (3, ) ->  (1, 3) -> (3, 3)

b = b[:, np.newaxis]
print(b, b.ndim, b.shape)
# 2 (3, 2) -> (3, 2)
# 2 (3, 1) -> (3, 2)
print(a + b)

## Q2: Пример для у. Вычислить количество элементов (по обоим размерностям)
# значение которых больше 3 и меньше 9

y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(np.sum((y > 3) & (y < 9)))
print(np.sum((y > 3) & (y < 9), axis=0))
print(np.sum((y > 3) & (y < 9), axis=1))

