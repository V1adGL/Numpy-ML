import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

iris = sns.load_dataset('iris')
data = iris[['petal_width', 'petal_length', 'sepal_length', 'species']]

# Отделим по видам
data_versicolor = data[data['species'] == 'versicolor'].drop(columns=['species'])
data_setosa = data[data['species'] == 'setosa'].drop(columns=['species'])

fig = go.Figure()

### === VERSICOLOR === ###
# Scatter
fig.add_trace(go.Scatter3d(
    x=data_versicolor['petal_width'],
    y=data_versicolor['petal_length'],
    z=data_versicolor['sepal_length'],
    mode='markers',
    marker=dict(size=5, color='green'),
    name='versicolor'
))

# PCA-3
p_v = PCA(n_components=3)
p_v.fit(data_versicolor)
mean_v = p_v.mean_
components_v = p_v.components_
explained_v = p_v.explained_variance_

# Центр
fig.add_trace(go.Scatter3d(
    x=[mean_v[0]], y=[mean_v[1]], z=[mean_v[2]],
    mode='markers',
    marker=dict(size=6, color='black', symbol='x'),
    name='Mean (versicolor)'
))

# Векторы
for i, color in enumerate(['red', 'orange', 'purple']):
    vec = components_v[i] * np.sqrt(explained_v[i])
    fig.add_trace(go.Scatter3d(
        x=[mean_v[0], mean_v[0] + vec[0]],
        y=[mean_v[1], mean_v[1] + vec[1]],
        z=[mean_v[2], mean_v[2] + vec[2]],
        mode='lines',
        line=dict(color=color, width=4),
        name=f'PCA {i+1} (versicolor)'
    ))

# PCA-2
p_v2 = PCA(n_components=2)
X_v2 = p_v2.fit_transform(data_versicolor)
X_v2_inv = p_v2.inverse_transform(X_v2)

fig.add_trace(go.Scatter3d(
    x=X_v2_inv[:, 0],
    y=X_v2_inv[:, 1],
    z=X_v2_inv[:, 2],
    mode='markers',
    marker=dict(size=4, color='red', opacity=0.8),
    name='Inverse 2D (versicolor)'
))


### === SETOSA === ###
# Scatter
fig.add_trace(go.Scatter3d(
    x=data_setosa['petal_width'],
    y=data_setosa['petal_length'],
    z=data_setosa['sepal_length'],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='setosa'
))

# PCA-3
p_s = PCA(n_components=3)
p_s.fit(data_setosa)
mean_s = p_s.mean_
components_s = p_s.components_
explained_s = p_s.explained_variance_

# Центр
fig.add_trace(go.Scatter3d(
    x=[mean_s[0]], y=[mean_s[1]], z=[mean_s[2]],
    mode='markers',
    marker=dict(size=6, color='black', symbol='x'),
    name='Mean (setosa)'
))

# Векторы
for i, color in enumerate(['red', 'orange', 'purple']):
    vec = components_s[i] * np.sqrt(explained_s[i])
    fig.add_trace(go.Scatter3d(
        x=[mean_s[0], mean_s[0] + vec[0]],
        y=[mean_s[1], mean_s[1] + vec[1]],
        z=[mean_s[2], mean_s[2] + vec[2]],
        mode='lines',
        line=dict(color=color, width=4),
        name=f'PCA {i+1} (setosa)'
    ))

# PCA-2
p_s2 = PCA(n_components=2)
X_s2 = p_s2.fit_transform(data_setosa)
X_s2_inv = p_s2.inverse_transform(X_s2)

fig.add_trace(go.Scatter3d(
    x=X_s2_inv[:, 0],
    y=X_s2_inv[:, 1],
    z=X_s2_inv[:, 2],
    mode='markers',
    marker=dict(size=4, color='red', opacity=0.8),
    name='Inverse 2D (setosa)'
))


### === Layout === ###
fig.update_layout(
    scene=dict(
        xaxis_title='Petal Width',
        yaxis_title='Petal Length',
        zaxis_title='Sepal Length'
    ),
    title='Метод главных компонент для setosa, versicolor',
    legend=dict(x=0.01, y=0.99)
)

fig.show()
