import numpy as np
import pandas as pd

# 1. Привести различные способы создания объектов типа Series
# Для создания Series можно использовать
# - списки Python или массивы NumPy
# - скалярные значение
# - словари

data1 = pd.Series([0, 2, 4, 6, 8], index = ['a', 'b', 'c', 'd', 'e'])
print(data1)

data2 = pd.Series({'a': 2, 'b': 3, 'c': 4, 'd': 5})
print(data2)

lst = np.arange(0, 10)
data3 = pd.Series(lst)
print(data3)

data4 = pd.Series(5.0, index=["a", "b", "c", "d", "e"])
print(data4)

# 2. Привести различные способы создания объектов типа DataFrame
# DataFrame. Способы создания

print('--------------------------')

# - через объекты Series
series_a = pd.Series(['A1', 'A2', 'A3'], name='row_A')
series_b = pd.Series(['B1', 'B2', 'B3'], name='row_B')
series_c = pd.Series(['C1', 'C2', 'C3'], name='row_C')

df = pd.DataFrame([series_a, series_b, series_c])
print(df)

# - словари объектов Series
data5 = pd.DataFrame({'data3': data3, 'data2': data2})
print(data5)

# - списки словарей
lst_dict = [
    {'name': 'Ivan', 'age': 33, 'profession': 'dentist'},
    {'name': 'Oleg', 'age': 22, 'profession': 'student'}
    ]
data6 = pd.DataFrame(lst_dict)
print(data6)

# - двумерный массив NumPy
lst = np.array([
    [0, 0, 0],
    [0, 1, 2],
    [0, 2, 4]
])
data8 = pd.DataFrame(lst)
print(data8)


# - структурированный массив Numpy
structured = np.array([(1, 'First', 0.5, 1+2j),
                       (2, 'Second', 1.3, 2-2j),
                       (3, 'Third', 0.8, 1+3j)],
                      dtype=[
                          ('id','i2'),
                          ('position','S6'),
                          ('value','f4'),
                          ('complex','c8')
                      ]
                      )
data9 = pd.DataFrame(structured)
print(data9)

# 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1

print('--------------------------')
series_a = pd.Series(np.arange(12, 15), index=np.arange(0, 3))
series_b = pd.Series(np.arange(15, 18), index=np.arange(3, 6))

df = pd.DataFrame({'A': series_a, 'B': series_b})
print(df.fillna(1))

# 4. Переписать пример с транслированием для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ

print('--------------------------')
rng = np.random.default_rng()
A = rng.integers(0,10, (3,4))

df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print(df)
print(df['a'])
print(df.subtract(df['a'], axis=0))

# 5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()

print('--------------------------')
df = pd. DataFrame(
    [[np.nan, 2, np.nan, 0],
         [3, 4, np.nan, 1],
         [np.nan, np.nan, np.nan, np.nan],
         [np.nan, 3, np.nan, 4]],
    columns=list("ABCD"))

print(df)
print(df.ffill())   # использует предыдущее известное наблюдение для заполнения пробела
# print(df.fillna(method="ffill"))   # еще один вариант записи. Устарел!

print(df)
print(df.bfill())   # использует следующее известное наблюдение для заполнения пробела
