# Деревья решений и случайные леса
# СЛ - непараметрический алгоритм
# СЛ - пример ансамблевого метода, основанного на агрегации результатов множества простых моделей
# В реализациях дерева принятия рещений в мащинном обучении, вопросы обычно ведут к разделению данных по осям, т.е.
# каждый узел разбивает данные на две группы по одному из признаков

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')
print(iris.head())

species_int = []

for r in iris.values:
    match r[4]:
        case 'setosa':
            species_int.append(1)
        case 'versicolor':
            species_int.append(2)
        case 'virginica':
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int_df

data_df = data[(data['species'] == 3) | (data['species'] == 2)]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 3]
data_df_versicolor = data_df[data_df['species'] == 2]

max_depth = [[1, 2, 3, 4], [5, 6, 7, 8]]
figure, ax = plt.subplots(2, 4, sharex='col', sharey='row')

for i in range(2):
    for j in range(4):
        ax[i, j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
        ax[i, j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])
        ax[i, j].set_title(f'Max depth = {max_depth[i][j]}')
        ax[i, j].set_xlim(4.5, 8)
        ax[i, j].set_ylim(2.5, 7)

        model = DecisionTreeClassifier(max_depth=max_depth[i][j])
        model.fit(X, y)

        x1_p = np.linspace(4.5, 8, 100)
        x2_p = np.linspace(2.5, 7, 100)
        # x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
        # x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

        X_p = pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
        )

        y_p = model.predict(X_p)

        ax[i, j].contourf(
            X1_p,
            X2_p,
            y_p.reshape(X1_p.shape),
            alpha=0.2,
            levels=2,
            cmap='rainbow',
            zorder=1)

plt.show()