import numpy as np
import array
import sys
import random

## 1. Какие еще существуют коды типов?
'''
typecode      Python Type
b               int
B               int
u               Unicode char
w               Unicode char
h               int
H               int
i               int
I               int
l               int
L               int
q               int
Q               int
f               float
d               float
'''

## 2. Напишите код, подобный приведенному выше, но с другим типом
print('\n Exercise 2')
a1 = array.array('b', b'abcdef')
print(sys.getsizeof(a1), a1)
print(type(a1))

a2 = array.array('f', [1, 2, 4, 6])
print(a2, type(a2))

## 3. Напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1

print('\n Exercise 3')
arr = np.linspace(0, 1, 5)
print(arr)

## 4. Напишите код для создания массива с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1

print('\n Exercise 4')
arr = np.array([random.random() for i in range(5)])
print(arr)

## 5. Напишите код для создания массива с 5 нормально распределенными случайными значениями с мат.ожиданием = 0 и дисперсией 1

print('\n Exercise 5')
arr = np.random.normal(0, 1, 5)
print(arr)

## 6. Напишите код для создания массива с 5 случайными целыми числами в от [0, 10)

print('\n Exercise 6')
arr = np.array([random.randrange(10) for _ in range(5)])
print(arr)

## 7. Написать код для создания срезов массива 3 на 4

print('\n Exercise 7')
massive = [[2, 3, 4, 5],
           [6, 7, 8, 9],
           [10, 11, 12 , 13]]

## - первые две строки и три столбца

res = [row[:3] for row in massive[:2]]
print(res)

## - первые три строки и второй столбец (разве это не то же самое, что и обычный столбец 2?)

res = [row[1] for row in massive[:]]
print(res)

## - все строки и столбцы в обратном порядке

res = [row[::-1] for row in massive[::-1]]
print(res)

## - второй столбец (вопрос выше)

res = [row[1] for row in massive[:]]
print(res)

## - третья строка

print(massive[2])

## 8. Продемонстрируйте, как сделать срез-копию

print('\n Exercise 8')

massive = [[2, 3, 4, 5],
           [6, 7, 8, 9],
           [10, 11, 12 , 13]]

copy = massive[:]   # Создаем срез-копию
print(copy)

# Проверка
copy[0] = 30
print(copy)
print(massive)   # Видно, что при изменении среза-копии copy, оригинал massive не меняется

## 9. Продемонстрируйте использование newaxis для получения вектора-столбца и вектора-строки

print('\n Exercise 9')

x = np.array([0, 1, 2, 3])
print(x)
row = x[np.newaxis, :]
print(row)

col = x[:, np.newaxis]
print(col)

## 10. Разберитесь, как работает метод dstack

print('\n Exercise 10')

x = np.array((3, 5, 7))
y = np.array((5, 7, 9))
print(np.dstack((x, y)))

x = np.array([[1], [2], [3]])
y = np.array([[2], [3], [4]])
print(np.dstack((x, y)))



## 11. Разберитесь, как работают методы split, vsplit, hsplit, dsplit

print('\n Exercise 11')

str = 'Hello world! That`s an example of split() operation.'
print(str.split(' '))

arr = np.arange(16).reshape(4, 4)
print(arr)

# Разбиение массива на 4 части по вертикали на равные части
print('Разбиение по вертикали: ', np.vsplit(arr, 4))

# Разбиение массива на 4 части по горизонтали (по столбцам) на равные части
print('Разбиение по горизонтали: ', np.hsplit(arr, 4))

arr = arr.reshape(2, 2, 4)
print(arr)

print('Разбиение по 3 оси: ', np.dsplit(arr, 4))

## 12. Привести пример использования всех универсальных функций, которые я привел
print('\n Exercise 12')

x = np.arange(10)
print('Стартовый массив: \n', x)

x1 = np.multiply(x, 2)
print('Умножение на 2: \n', x1)

x2 = np.add(x1, 2)
print('Добавляем 2: \n', x2)

x3 = np.divide(x2, 2)
print('Деление на 2 : \n', x3)

x4 = np.power(x3, 2)
print('Возведение в степень 2 : \n', x4)

x5 = np.floor_divide(x4, 4)
print('Целочисленное деление на 4 : \n', x5)

x6 = np.mod(x5, 6)
print('Остаток от деления на 6 : \n', x6)

x7 = np.subtract(x6, 3)
print('Вычитание 3 : \n', x7)

a = np.arange(7)
print(a, '\n Унарный минус \n',np.negative(a))
