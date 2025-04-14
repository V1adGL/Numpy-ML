# Наивная байесовская классификация
# Набор моделей, которые предлагают быстрые и простые алгоритмы классификации
# Очень хорошо подходит при большом колличестве признаков
# Часто используется для получения первого приближенного решения задачи классификации

# Теорема Байеса
#      P(A|B) = P(B|A) * P(A) / P(B)

#      P(A|B) - вероятность гипотезы А при наступлении события В == апостериорная вероятность (после события)
#      P(B|A) - вероятность события В при истинности гипотезы A == априорная вероятность (перед событием)
#      P(В) - полная вероятность наступления события В
#      P(В) = sum( P(B|A_i) * P(A_i) )


# Пример
# Пусть есть 2 6-ти гранных кубика
# K1 = {1, 2, 3, 4, 5, 6}
# K2 = {1, 2, 3, 4, 5, 1}
# Гипотеза (А): выбран K1 | выбран K2
# Событие (В): после броска выпала 1, 2, 3, 4, 5, 6

# P(K1|6) = 1/6 * 1/2 / (1/12) = 1          P(K2|6) = 0 * 1/2 / (1/12) = 0
# P(K1|2) =  1/6 * 1/2 / (1/6) = 1/2          P(K2|2) = 1/6 * 1/2 / (1/6) = 1/2
# P(K1|1) = 1/6 * 1/2 / (1/4) = 1/3         P(K2|1) = 2/6 * 1/2 / (1/4) = 2/3
# P(B) = P(B|K1) * P(K1) + P(B|K2) * P(K2) =

# P(6) = 1/6 * 1/2 + 0 * 1/2 = 1/12
# P(2) = 1/6 * 1/2 + 1/6 * 1/2 = 1/6
# P(1) = 1/6 * 1/2 + 2/6 * 1/2 = 1/4

#             P(B|A) * P(A)
# P(A|B) = -------------------
#                 P(B)
####################################################
#                   P(признаки|L) * P(L)
# P(L|признаки) = -----------------------
#                       P(признаки)
####################################################

# У нас бинарная классификация L1 или L2


#                   P(признаки|L1) * P(L1)
# P(L1|признаки) = -----------------------
#                       P(признаки)
####################################################
#                   P(признаки|L2) * P(L2)
# P(L2|признаки) = -----------------------
#                       P(признаки)

# Такая модель называется генеративной моделью

# Наивное допущение относительно генеративной модели => грубое приближение для каждого класса
# Гауссовский наивный байесовский классификатор
# Допущение состоит в том, что ! данные всех категорий взяты из простого нормального распределения !

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
iris = sns.load_dataset('iris')
print(iris.head())

# sns.pairplot(iris, hue='species')

data = iris[['sepal_length', 'petal_length', 'species']]
print(data.head())
print(data.shape)

#### setosa versicolor

data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]
print(data_df.shape)

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

model = GaussianNB()
model.fit(X, y)

print(model.theta_[0])
print(model.var_[0])
print(model.theta_[1])
print(model.var_[1])

theta0 = model.theta_[0]
var0 = model.var_[0]
theta1 = model.theta_[1]
var1 = model.var_[1]

data_df_setosa  = data_df[data_df['species'] == 'setosa']
data_df_versicolor =data_df[data_df['species'] == 'versicolor']


plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

plt.title("Setosa vs Versicolor (2D)")


x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)


X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)

print(X_p.head())

z1 = (
        1 / (1 * np.pi * (var0[0] * var0[1]) ** 0.5) *
      np.exp(-0.5 * ((X1_p - theta0[0]) ** 2 / (var0[0]) + (X2_p - theta0[1]) ** 2 / (var0[1]))
      )
)
plt.contour(X1_p, X2_p, z1)

z2 = (
        1 / (1 * np.pi * (var1[0] * var1[1]) ** 0.5) *
      np.exp(-0.5 * ((X1_p - theta1[0]) ** 2 / (var1[0]) + (X2_p - theta1[1]) ** 2 / (var1[1]))
      )
)
plt.contour(X1_p, X2_p, z2)

y_p = model.predict(X_p)

X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

print(X_p.head())

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)


fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(X1_p, X2_p, z1, 40)
ax.contour3D(X1_p, X2_p, z2, 40)

ax.set_title("Setosa vs Versicolor (3D)")
plt.show()

#### virginica versicolor

# data_df2 = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]
# print(data_df2.shape)
#
# X = data_df2[['sepal_length', 'petal_length']]
# y = data_df2['species']
#
# model = GaussianNB()
# model.fit(X, y)
#
# print(model.theta_[0])
# print(model.var_[0])
# print(model.theta_[1])
# print(model.var_[1])
#
# theta0 = model.theta_[0]
# var0 = model.var_[0]
# theta1 = model.theta_[1]
# var1 = model.var_[1]
#
# data_df_virginica = data_df2[data_df2['species'] == 'virginica']
# data_df_versicolor2 = data_df2[data_df2['species'] == 'versicolor']
#
# fig1 = plt.figure()
# plt.scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
# plt.scatter(data_df_versicolor2['sepal_length'], data_df_versicolor2['petal_length'])
#
# plt.title("Virginica vs Versicolor (2D)")
# plt.show()
#
# x1_p = np.linspace(min(data_df2['sepal_length']), max(data_df2['sepal_length']), 100)
# x2_p = np.linspace(min(data_df2['petal_length']), max(data_df2['petal_length']), 100)
#
# X1_p, X2_p = np.meshgrid(x1_p, x2_p)
#
#
# X_p = pd.DataFrame(
#     np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
# )
#
# print(X_p.head())
#
# z1 = (
#         1 / (1 * np.pi * (var0[0] * var0[1]) ** 0.5) *
#       np.exp(-0.5 * ((X1_p - theta0[0]) ** 2 / (var0[0]) + (X2_p - theta0[1]) ** 2 / (var0[1]))
#       )
# )
# plt.contour(X1_p, X2_p, z1)
#
# z2 = (
#         1 / (1 * np.pi * (var1[0] * var1[1]) ** 0.5) *
#       np.exp(-0.5 * ((X1_p - theta1[0]) ** 2 / (var1[0]) + (X2_p - theta1[1]) ** 2 / (var1[1]))
#       )
# )
# plt.contour(X1_p, X2_p, z2)
#
# y_p = model.predict(X_p)
#
# X_p['species'] = y_p
#
# X_p_virginica = X_p[X_p['species'] == 'virginica']
# X_p_versicolor2 = X_p[X_p['species'] == 'versicolor']
#
# print(X_p.head())
#
# plt.scatter(X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha=0.2)
# plt.scatter(X_p_versicolor2['sepal_length'], X_p_versicolor2['petal_length'], alpha=0.2)
#
#
# fig2 = plt.figure()
#
# ax2 = plt.axes(projection='3d')
#
# ax2.contour3D(X1_p, X2_p, z1, 40)
# ax2.contour3D(X1_p, X2_p, z2, 40)
#
# ax2.set_title("Virginica vs Versicolor (3D)")
plt.show()


