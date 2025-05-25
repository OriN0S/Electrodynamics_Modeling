import numpy as np
import matplotlib.pyplot as plt

#Вариант 13
#Функция, вычисляющая значение выражения из задания
def function(x, D = 2):
    y2 = 0
    for i in range(D):
        y2 += x[i] * np.sin(np.sqrt(np.fabs(x[i])))
    y = 418.9829 * D + y2
    return y

#Функция для рассчёта выражения от одной переменной
def graph_2d(xmin, xmax, x10 = None, x20 = None, count = 200):
    xgrid = [0, 0]

    #Проверка на функцию от x1 при x2 = x20
    if x10 == None:
        x = np.linspace(xmin, xmax, count)

        xgrid[0] = x
        xgrid[1] = x20

        y = function(xgrid)
        return xgrid[0], y
    
    #Проверка на функцию от x2 при x1 = x10
    elif x20 == None:
        x = np.linspace(xmin, xmax, count)

        xgrid[0] = x10
        xgrid[1] = x

        y = function(xgrid)
        return xgrid[1], y
    
    else:
        return None, None

#Функция для рассчёта выражения от двух переменных
def graph_3d(x1min, x1max, x2min, x2max, count = 200):
    xgrid = [0, 0]

    x1 = np.linspace(x1min, x1max, count)
    x2 = np.linspace(x2min, x2max, count)

    #Создаем двумерную матрицу-сетку
    xgrid[0], xgrid[1] = np.meshgrid(x1, x2)

    y = function(xgrid)
    return xgrid[0], xgrid[1], y

#Функция для построения графиков
def draw(x1min, x1max, x2min, x2max):
    fig = plt.figure(figsize=[16, 9])

    yn = function([x['10'], x['20']])

    x1, x2, y = graph_3d(x1min, x1max, x2min, x2max)
    
    #Задаю позицию графика в едином окне и его размерность
    axes_1 = fig.add_subplot(2, 2, 1, projection='3d')
    axes_1.set(xlabel="x\u2081", ylabel="x\u2082", zlabel="y = f(x\u2081, x\u2082)")
    axes_1.plot_surface(x1, x2, y, cmap="hot")
    #Отмечаю точку с заданными координатами
    axes_1.scatter(x['10'], x['20'], yn, color='black', marker='o')
    #Подписываю точку с заданными координатами
    axes_1.text(x['10'], x['20'], yn, "y = f(x\u2081\u2080, x\u2082\u2080)")

    #Задаю позицию графика в едином окне и его размерность
    axes_2 = fig.add_subplot(2, 2, 2, projection='3d')
    axes_2.set(xlabel="x\u2081", ylabel="x\u2082", zlabel="y = f(x\u2081, x\u2082)")
    #Задаю точку наблюдения
    axes_2.view_init(elev=90, azim=90)
    axes_2.plot_surface(x1, x2, y, cmap="hot")
    #Отмечаю точку с заданными координатами
    axes_2.scatter(x['10'], x['20'], yn, color='black', marker='o')
    #Подписываю точку с заданными координатами
    axes_2.text(x['10'], x['20'], yn, "y = f(x\u2081\u2080, x\u2082\u2080)")
    x1, y = graph_2d(x1min, x1max, x20 = x['20'])

    #Задаю позицию графика в едином окне
    axes_3 = fig.add_subplot(2, 2, 3)
    axes_3.set(xlabel = ("x\u2081"), ylabel = ("y = f(x\u2081)"))
    axes_3.plot(x1, y, color = "#735184")
    #Отмечаю точку с заданными координатами
    axes_3.scatter(x['10'], yn, color='black', marker='o')
    #Подписываю точку с заданными координатами
    axes_3.text(x['10'], yn, "y = f(x\u2081\u2080)")

    x2, y = graph_2d(x2min, x2max, x10 = x['10'])

    #Задаю позицию графика в едином окне
    axes_4 = fig.add_subplot(2, 2, 4)
    axes_4.set(xlabel = ("x\u2082"), ylabel = ("y = f(x\u2082)"))
    axes_4.plot(x2, y, color = "#735184")
    #Отмечаю точку с заданными координатами
    axes_4.scatter(x['20'], yn, color='black', marker='o')
    #Подписываю точку с заданными координатами
    axes_4.text(x['20'], yn, "y = f(x\u2082\u2080)")

    plt.show()

if __name__ == "__main__":
    x1min = -500.
    x1max = 500.
    x2min = -500.
    x2max = 500.
    x = {'10': 420.9687, '20': 420.9687}

    draw(x1min, x1max, x2min, x2max)
