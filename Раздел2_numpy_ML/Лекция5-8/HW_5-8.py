import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

######### Plot 1 ##############

x = [1, 5, 10, 15, 20]
y1 = [0.9, 7, 3.75, 5.5, 11]
y2 = [4, 3, 1, 8.2, 12]

fig = plt.figure()
plt.plot(x, y1, 'o', linestyle='-', color='red')
plt.plot(x, y2, 'o', linestyle='-.', color='green')
plt.legend(['line 1', 'line 2'], loc='upper left')

########## Plot 2 ###############

fig2 = plt.figure()
grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.4)

x1 = [1, 2, 3, 4, 5]
y1 = [0.5, 7, 6, 3, 5]
plt.subplot(grid[0, :4])
plt.plot(x1, y1)

y2 = [(i - 3) ** 2 for i in x1]
plt.subplot(grid[1, :2])
plt.plot(x1, y2)

y3 = [-7, -4, 2, -4, -7]
plt.subplot(grid[1, 2:4])
plt.plot(x1, y3)



########## Plot 3 ###############

fig3, ax = plt.subplots()
x = np.linspace(-5, 5, 11)
y = [i ** 2 for i in x]
ax.plot(x, y)
ax.annotate('min', xy=(0, 0), xytext=(0, 10), arrowprops=dict(facecolor='green'))


######### Plot 4 ###############

fig4, ax = plt.subplots()

data = np.random.rand(8, 8) * 10
df = pd.DataFrame(data)

custom_xticks = ['7', '6', '5', '4', '3', '2', '1', '0']

sns.heatmap(data, annot=False, fmt=".1f", cbar=True, cmap='viridis', vmin=0, vmax=10, yticklabels=custom_xticks)


########### Plot 5 ###################

fig5, ax = plt.subplots()

x = np.linspace(0, 5, 1000)
y = np.cos(np.pi * x)
plt.plot(x, y, color='red')
plt.fill_between(x, y)

########### Plot 6 ###################

fig6, ax = plt.subplots()

def f(x):
    lst = []
    for i in x:
        if np.cos(np.pi * i) > -0.5:
            lst.append(np.cos(np.pi * i))
        else:
            lst.append(None)
    return lst

x = np.linspace(0, 5, 1000)
y = f(x)

plt.plot(x, y)
ax.set(xlim=(-0.2, 4.95), ylim=(-1, 1))

########### Plot 7 ###################

def configure_axes(ax, x, y, where):
    ax.step(x, y, color='green', where=where)
    ax.scatter(x, y, color='green', marker='o')
    ax.set_aspect('equal')  # Квадратные оси
    ax.grid(True, which='both')  # Включаем сетку
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.2, 6.2)
    ax.set_xticks(np.arange(0, 7, 1))  # Шаг сетки по оси X
    ax.set_yticks(np.arange(0, 7, 1))  # Шаг сетки по оси Y

fig7 = plt.figure(figsize=(9, 3))
gs = plt.GridSpec(1, 3)

ax1 = fig7.add_subplot(gs[0, 0])
ax2 = fig7.add_subplot(gs[0, 1])
ax3 = fig7.add_subplot(gs[0, 2])

x = np.linspace(0, 6, 7)
y = x

configure_axes(ax1, x, y, 'pre')
configure_axes(ax2, x, y, 'post')
configure_axes(ax3, x, y, 'mid')

plt.tight_layout()


########### Plot 8 ###################

fig8, ax = plt.subplots()

x = np.linspace(0, 10, 10)
y1 = 5 * np.sin(np.pi / 10 * x)
y2 = 10 * np.sin(np.pi / 10 * x)
y3 = 26 * np.sin(np.pi / 14 * x)


ax.fill_between(x, y1, label='y1')
ax.fill_between(x, y1, y2, color='orange', label='y2')
ax.fill_between(x, y2, y3, color='green', label='y3')

ax.legend()


########### Plot 8 ###################

fig8 = plt.figure()
data = {'Auto': ['BMW', 'AUDI', 'Jaguar', 'Ford', 'Toyota'],
        'Values': [33, 13, 22, 17, 10]}
df = pd.DataFrame(data)


explode = [0.1, 0, 0, 0, 0]   # смещение первого сектора
colors = ['g', 'red', 'mediumpurple', 'royalblue', 'darkorange']

plt.pie(df['Values'], labels=df['Auto'], startangle=100, colors=colors, explode=explode)

plt.axis('equal')


########### Plot 9 ###################

fig9 = plt.figure()
data = {'Auto': ['BMW', 'AUDI', 'Jaguar', 'Ford', 'Toyota'],
        'Values': [33, 13, 22, 17, 10]}
df = pd.DataFrame(data)


colors = ['g', 'red', 'mediumpurple', 'royalblue', 'darkorange']

plt.pie(df['Values'], labels=df['Auto'], startangle=100, colors=colors,
        wedgeprops={'width': 0.5})

plt.axis('equal')

plt.show()