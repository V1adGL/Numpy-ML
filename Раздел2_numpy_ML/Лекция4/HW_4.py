import pandas as pd
import numpy as np

# # 1. Разобраться как использовать мультииндексные ключи в данном примере
index = pd.MultiIndex.from_tuples([
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
])

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]
pop = pd.Series(population, index=index)
pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)

print(pop_df)

print('--------------')
# Ошибка была в том, что был неправильно создан мультииндекс (скорее всего)
# Как вариант решения - создать мультииндекс явно через pd.MultiIndex()
pop_df_1 = pop_df.loc['city_1', 'something']
print(pop_df_1)

print('--------------')

pop_df_2 = pop_df.loc[['city_1', 'city_3'], ['total', 'something']]
print(pop_df_2)

print('--------------')

pop_df_3 = pop_df.loc[['city_1', 'city_3'], 'something']
print(pop_df_3)

print('--------------')


# 2. Из получившихся данных выбрать данные по
# - 2020 году (для всех столбцов)
# - job_1 (для всех строк)
# - для city_1 и job_2
index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)
print('--------------')

print(index)

print('--------------')

columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)

print(columns)

print('--------------')

rng = np.random.default_rng(1)

data = rng.random((4, 6))

print(data)

print('--------------')

data_df = pd.DataFrame(data, index=index, columns=columns)
print(data_df)

print('--------------')
# - 2020 году (для всех столбцов)
print('2020 году (для всех столбцов) \n'
      '-----------Вариант 1----------')
data_df_1 = data_df.loc[(slice(None), 2020), :]
print(data_df_1)

print('-----------Вариант 2----------')
data_df_1 = data_df.loc[(('city_1', 'city_2'), 2020), :]
print(data_df_1)


print('--------------')

# - job_1 (для всех строк)
print('job_1 (для всех строк) \n'
      '-----------Вариант 1----------')
data_df_2 = data_df.loc[(slice(None), slice(None)), (slice(None), 'job_1')]
print(data_df_2)

print('-----------Вариант 2----------')

data_df_2 = data_df.loc[:, pd.IndexSlice[:, 'job_1']]
print(data_df_2)

print('-----------Вариант 3----------')
data_df_2 = data_df.xs('job_1', level='job', axis=1)
print(data_df_2)

# - для city_1 и job_2

print(' для city_1 и job_2 (для всех строк) \n'
      '-----------Вариант 1----------')

data_df_3 = data_df.loc[('city_1', slice(None)), (slice(None), 'job_2')]
print(data_df_3)

print('-----------Вариант 2----------')

data_df_3 = data_df.loc[('city_1', slice(None)), pd.IndexSlice[:, 'job_2']]
print(data_df_3)


# 3. Взять за основу DataFrame со следующей структурой
index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)
columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)

rng = np.random.default_rng(1)
data = rng.random((4, 6))
data_df = pd.DataFrame(data, index=index, columns=columns)

print('----------------------------------')
print(data_df)

# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
print('-------------Вариант 1--------------------')
data_df_1 = data_df.loc[:, ((['person_1', 'person_3']), ('job_1', 'job_2'))]
print(data_df_1)

print('-------------Вариант 2--------------------')

data_df_1 = data_df.loc[:, ['person_1', 'person_3']]
print(data_df_1)
# - все данные по первому городу и первым двум person-ам (с использованием срезов)

print('-------------Вариант 1--------------------')

data_df_2 = data_df.loc[('city_1', slice(None)), ['person_1', 'person_2']]
print(data_df_2)

# Приведите пример (самостоятельно) с использованием pd.IndexSlice

data_df_2 = data_df.loc[pd.IndexSlice[['city_2'], [2020]], pd.IndexSlice[['person_2']]]
print(data_df_2)


#4. Привести пример использования inner и outer джойнов для Series (данные примера скорее всего нужно изменить)
ser1 = pd.Series(['a', 'b', 'c'], index=[100, 20, 30],)
ser2 = pd.Series(['b', 'c', 'd'], index=[100, 200, 300])

# Попробовал поменять местами индексы и массивы - заработало

# join='inner' оставляет только те индексы, которые присутствуют в обеих Series
print (pd.concat([ser1, ser2], join='inner', axis=1))

# join='outer' сохраняет все индексы из обеих Series, вставляя NaN для отсутствующих значений
print (pd.concat([ser1, ser2], join='outer', axis=1))