import numpy as np
import pandas as pd

# Pandas - расширение numpy (структурированные массивы)
# Строки и столбцы инзе

# Series, DataFrame, Index

## Series

# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data)
# print(type(data))
#
# print(data.values)
# print(type(data.values))
#
# print(data.index)
# print(type(data.index))

# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data[0])
# print(data[1:3])

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['1', 10, 7, 'd'])   # Можно использовать разные типы для индексов
#
# print(data)
# print(data['1'])
# print(data[10:'d'])
#
# print(type(data.index))

# population_dict = {
#     'city1': 1000,
#     'city2': 1001,
#     'city3': 1002,
#     'city4': 1003,
#     'city5': 1004
# }
# population = pd.Series(population_dict)
# print(population)
# print(population['city4'])
# print(population['city4':'city5'])
# Для создания Series можон использовать
# списки питона или массивы numpy
# скалярные значения
# словари
# Q1: ________-

## DataFrame - двумерный масссив с явно определенными индексами. Последовательность
# согласованных по индексам объектов Series

# population_dict = {
#     'city1': 1000,
#     'city2': 1001,
#     'city3': 1002,
#     'city4': 1003,
#     'city5': 1004
# }
# area_dict = {
#     'city1': 9990,
#     'city2': 9991,
#     'city3': 9992,
#     'city4': 9993,
#     'city5': 9994
# }
# population = pd.Series(population_dict)
# area = pd.Series(area_dict)
#
# states = pd.DataFrame({
#     'population1': population,
#     'area1': area
# })
# print(states)

# print(states.values)
# print(states.index)
# print(states.columns)
#
# print(type(states.values))
# print(type(states.index))
# print(type(states.columns))

# print(states['area1'])

# DataFrame. Способы создания
# - через объекы series
# - списки соловарей
# - словари объектов Series
# - двумерный массив NumPy
# - структурированный массив NumPy
## Q2 ___________-

## Index - способ организации ссылки на данные объектов Series and DataFrame
# он неизменяем, упорядочен, является мультимножествов (есть повторные значения)

# ind = pd.Index

## Выборка данных из Series
# (ведет себя как словарь)

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
# print(data)
# print('a' in data)
# print('z' in data)
#
# print(data.keys())
# print(list(data.items()))
#
# data['a'] = 100
# data['z'] = 1000
# print(data)
#
# # как одномерный массив
#
# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
#
# print(data['a':'c'])
# print(data[0:2])
# print(data[(data > 0.5) & (data < 1)])
# print(data[['a', 'd']])
#
# # атрибуты-индексаторы
# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 3, 10, 15])
#
# print(data[1])
#
# print(data.loc[1])   # наши значения
# print(data.iloc[1])   # индексы

## Выборка данных из DataFrame
# like Dict
# population_dict = {
#     'city1': 1000,
#     'city2': 1001,
#     'city3': 1002,
#     'city4': 1003,
#     'city5': 1004
# }
# area_dict = {
#     'city1': 9990,
#     'city2': 9991,
#     'city3': 9992,
#     'city4': 9993,
#     'city5': 9994
# }
#
# pop = pd.Series(
#     {
#     'city1': 1000,
#     'city2': 1001,
#     'city3': 1002,
#     'city4': 1003,
#     'city5': 1004
#     })
#
# pop1 = pd.Series(
#     {
#     'city1': 1000,
#     'city2': 1001,
#     'city3': 1002,
#     'city4': 1003,
#     'city5': 1004
#     })
#
# area = pd.Series(
#     {
#     'city1': 9990,
#     'city2': 9991,
#     'city3': 9992,
#     'city4': 9993,
#     'city5': 9994
#     })
#
#
# data = pd.DataFrame({'area1': area, 'pop1': pop, 'pop': pop})
#
# print(data)
# print(data['area1'])
# print(data.area1)   # может возникнуть проблема (
#
# print(data.pop1 is data['pop1'])
# print(data.pop is data['pop'])
#
# data['new'] = data['area1']
#
# data['new1'] = data['area1'] / data['pop1']
#
# print(data)

# двумерный Numpy - массив

# data = pd.DataFrame({'area1': area, 'pop1': pop, 'pop': pop})

# print(data)
#
# print(data.values)
#
# print(data.T)
#
# print(data['area1'])
#
# print(data.values[0:3])

# атрибуты - индексаторы

# print(data)
# print(data.iloc[:3, 1:2])
# print(data.loc[:"city4", 'pop1':'pop'])
# print(data.loc[data['pop'] > 1002, ['area1','pop']])
#
# data.iloc[0,2] = 999999
# print(data)


# rng = np.random.default_rng()
# # s = pd.Series(rng.integers(0, 10, 4))
# # print(s)
# # print(np.exp(s))
#
#
# pop = pd.Series(
#     {
#     'city1': 1000,
#     'city2': 1001,
#     'city3': 1002,
#     'city4': 1003,
#     'city5': 1004
#     })
#
# pop1 = pd.Series(
#     {
#     'city1': 1000,
#     'city2': 1001,
#     'city3': 1002,
#     'city4': 1003,
#     'city5': 1004
#     })
#
# area = pd.Series(
#     {
#     'city1': 9990,
#     'city2': 9991,
#     'city3': 9992,
#     'city44': 9993,
#     'city55': 9994
#     })
#
# data = pd.DataFrame({'area1': area, 'pop1': pop, 'pop': pop})
#
# print(data)

# NaN = not a number (индексы одного отсутсвуют в другой, следовательно добавляются отсутсвующие

## Q3 _________-

## Объединение DataFrame

# dfa = pd.DataFrame(rng.integers(0, 10, (2,2)), columns=['a','b'])
# dfb = pd.DataFrame(rng.integers(0, 10, (3,3)), columns=['a','b', 'c'])
#
# print(dfa)
# print(dfb)
# print(dfa + dfb)

rng = np.random.default_rng()

A = rng.integers(0,10, (3,4))
print(A)
# print(A[0])
# print(A - A[0])

df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print(df)
print(df.iloc[0])
print(df - df.iloc[0])   # вычитание через транслирование (разные размерности)

print(df.iloc[0, ::2])

print(df - df.iloc[0, ::2])

## Q4 __________-

## NA - значения (not avalable): Nan, null, -99999

# Pandas. Два способа хранения отсутсTвующих значений
# индикаторы Nan, None
# null

# None - объект (накладные расходы). Не работает с sum, min

val1 = np.array([1,2,3])
# val1 = np.array([1,Non,3])   # Error
print(val1.sum())

val1 = np.array([1,np.nan,3])
print(val1)
print(val1.sum())
print(np.sum(val1))
print(np.nansum(val1))

x = pd.Series(range(10), dtype=int)
print(x)

x[0] = None
x[1] = np.nan
print(x)

x1 = pd.Series(['a', 'b', 'c'])
print(x1)

x1[0] = None
x1[1] = np.nan
print(x1)


x2 = pd.Series([1,2,3,np.nan, None, pd.NA])
print(x2)

x3 = pd.Series([1,2,3,np.nan, None, pd.NA], dtype='Int32')
print(x3)

print(x3[x3.isnull()])
print(x3[x3.notnull()])

print(x3.dropna())

df = pd.DataFrame(
    [
    [1,2,3,np.nan, None, pd.NA],
    [1,2,3,None,5,6],
    [1, np.nan, 3, None, np.nan, 6]
    ])
print(df)
# print(df.dropna())
# print('dddddd')
# print(df.dropna(axis=0))
# print(df.dropna(axis=1))

## how
# - all - все значения NA
# - any - хотя бы одно
# - thresh = x, остается, если присутствует МИНИМУМ x НЕПУСТЫХ значений

print(df.dropna(axis=1, how='all'))
print(df.dropna(axis=1, how='any'))


print(df.dropna(axis=1, thresh=2))

## Q5 ____________- 