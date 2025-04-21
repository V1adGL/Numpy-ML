import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


iris = sns.load_dataset('iris')
data = iris[['sepal_length', 'petal_length', 'species']]

samples = [100, 75, 50]
figure, ax = plt.subplots(1, 3, sharex='col', sharey='row')

for i in range(3):

    data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]
    data_df = data_df.sample(samples[i])

    X = data_df[['sepal_length', 'petal_length']]
    y = data_df['species']

    data_df_virginica = data_df[data_df['species'] == 'virginica']
    data_df_versicolor = data_df[data_df['species'] == 'versicolor']


    ax[i].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
    ax[i].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])
    ax[i].set_title(f'Число данных в выборке: {samples[i]}')

    model = SVC(kernel='linear', C=1000)
    model.fit(X, y)

    ax[i].scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=400,
        facecolor='none',
        edgecolor='black'
    )

    x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
    x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

    X1_p, X2_p = np.meshgrid(x1_p, x2_p)

    X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
    )

    y_p = model.predict(X_p)

    X_p['species'] = y_p

    X_p_virginica = X_p[X_p['species'] == 'virginica']
    X_p_versicolor = X_p[X_p['species'] == 'versicolor']

    ax[i].scatter(X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha=0.03)
    ax[i].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.03)

plt.show()
