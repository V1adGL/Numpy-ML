import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
df = px.data.iris()

# Выберем из датасета 2 вида ирисов
df = df[df['species'] != 'setosa'].copy()

# Преобразуем строковые метки в числовые
le = LabelEncoder()
y = le.fit_transform(df['species'])

# Матрица признаков (чтобы сохранять порядок столбцов)
features = ['sepal_width', 'petal_length', 'petal_width']
X = df[features]

# Создаем модель SVM
model = SVC(kernel='linear', C=1000)
model.fit(X, y)

# Базовый график точек
fig = px.scatter_3d(
    df, x="petal_length", y="petal_width", z="sepal_width",
    color="species",
    title="Разделение классов методом опорных векторов"
)

# Сетка для решающей границы
x_p = np.linspace(df['petal_length'].min(), df['petal_length'].max(), 100)
y_p = np.linspace(df['petal_width'].min(), df['petal_width'].max(), 100)
z_p = np.linspace(df['sepal_width'].min(), df['sepal_width'].max(), 100)
X_p, Y_p, Z_p = np.meshgrid(x_p, y_p, z_p)

# Создаем одномерный массив из признаков
X_pred = pd.DataFrame(
        np.vstack([Z_p.ravel(), X_p.ravel(), Y_p.ravel()]).T, columns=features
    )

# Получаем предсказания
y_pred = model.predict(X_pred)
y_pred = y_pred.reshape(X_p.shape)

# Добавляем изоповерхность (границу решения)
fig.add_trace(
    go.Isosurface(
        x=X_p.flatten(),
        y=Y_p.flatten(),
        z=Z_p.flatten(),
        value=y_pred.flatten(),
        isomin=0.5,
        isomax=0.5,
        opacity=0.3,
        surface_count=1,
        colorscale=['red', 'blue'],
        name='Разделяющая плоскость',
        showscale=False,
        surface=dict(fill=0.5, pattern='odd'),  # улучшает сглаживание
        caps=dict(x_show=False, y_show=False, z_show=False)
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title='Petal Length',
        yaxis_title='Petal Width',
        zaxis_title='Sepal Width'
    )
)

fig.show()