import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')

df = iris[iris['species'].isin(['virginica', 'versicolor'])].copy()

features = ['petal_width', 'petal_length', 'sepal_length']
X = df[features]

# Применяем KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Добавим метки для сравнения (правильно сопоставляем)
df['true_label'] = df['species'].map({'versicolor': 0, 'virginica': 1})

# Сопоставляем кластеры с метками через majority vote
clusters = df['cluster'].to_numpy()
true_labels = df['true_label'].to_numpy()
labels_mapped = np.zeros_like(clusters)
for cluster in np.unique(clusters):
    mask = clusters == cluster
    majority_label = mode(true_labels[mask], keepdims=True)[0][0]
    labels_mapped[mask] = majority_label

# Подсчет точности и матрицы ошибок
acc = accuracy_score(true_labels, labels_mapped)
cm = confusion_matrix(true_labels, labels_mapped)

print(f"Точность кластеризации: {acc:.2f}")
print("Матрица ошибок:\n", cm)

# 3D график с кластерами и центроидами
fig = px.scatter_3d(
    df,
    x='petal_width',
    y='petal_length',
    z='sepal_length',
    color=df['cluster'].astype(str),
    symbol=df['true_label'].astype(str),
    title='K-Means кластеризиция (Virginica & Versicolor)',
    labels={"color": "Кластер", "symbol": "Метка"}
)

centroids = kmeans.cluster_centers_

fig.add_trace(go.Scatter3d(
    x=centroids[:, 0],
    y=centroids[:, 1],
    z=centroids[:, 2],
    mode='markers',
    marker=dict(size=10, color='black', symbol='x'),
    name='Центроиды'
))

fig.show()

# Визуализация confusion matrix через seaborn
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Предсказано 0', 'Предсказано 1'],
            yticklabels=['Истинно 0', 'Истинно 1'])
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.title('Confusion Matrix')
plt.show()