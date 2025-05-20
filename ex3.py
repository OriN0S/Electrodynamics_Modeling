import numpy as np
import matplotlib.pyplot as plt

#Вариант 13
def function(x, D = 2): #Функция, вычисляющее значение выражения из задания
    y2 = 0
    for i in range(D):
        y2 += x[i] * np.sin(np.sqrt(np.fabs(x[i])))
    y = 418.9829 * D + y2
    return y

def graph_2d(xmin, xmax, x10 = None, x20 = None, count = 200): #Функция для рассчёта выражения от одной переменной
    xgrid = [0, 0]

    if (x10 == None): #Проверка на функцию от x1 при x2 = x20
        x = np.linspace(xmin, xmax, count)

        xgrid[0] = x
        xgrid[1] = x20

        y = function(xgrid)
        return xgrid[0], y
    
    elif (x20 == None): #Проверка на функцию от x2 при x1 = x10
        x = np.linspace(xmin, xmax, count)

        xgrid[0] = x10
        xgrid[1] = x

        y = function(xgrid)
        return xgrid[1], y
    
    else:
        return None

def graph_3d(x1min, x1max, x2min, x2max, count = 200): #Функция для рассчёта выражения от двух переменных
    xgrid = [0, 0]

    x1 = np.linspace(x1min, x1max, count)
    x2 = np.linspace(x2min, x2max, count)

    xgrid[0], xgrid[1] = np.meshgrid(x1, x2) #Создаем двумерную матрицу-сетку

    y = function(xgrid)
    return xgrid[0], xgrid[1], y

def draw(x1min, x1max, x2min, x2max): #Функция для построения трехмерного графика
    fig = plt.figure(figsize=[16, 9])

    yn = function([x['10'], x['20']])

    x1, x2, y = graph_3d(x1min, x1max, x2min, x2max)
    
    axes_1 = fig.add_subplot(2, 2, 1, projection='3d') #Задаю позицию графика в едином окне и его размерность
    axes_1.set(xlabel="x\u2081", ylabel="x\u2082", zlabel="y = f(x\u2081, x\u2082)") #Задаю названия осей
    axes_1.plot_surface(x1, x2, y, cmap="hot")
    axes_1.scatter(x['10'], x['20'], yn, color='black', marker='o') #Отмечаю точку с заданными координатами
    axes_1.text(x['10'], x['20'], yn, "y = f(x\u2081\u2080, x\u2082\u2080)") #Подписываю точку с заданными координатами

    axes_2 = fig.add_subplot(2, 2, 2, projection='3d') #Задаю позицию графика в едином окне и его размерность
    axes_2.set(xlabel="x\u2081", ylabel="x\u2082", zlabel="y = f(x\u2081, x\u2082)") #Задаю названия осей
    axes_2.view_init(elev=90, azim=90) #Задаю точку наблюдения
    axes_2.plot_surface(x1, x2, y, cmap="hot")
    axes_2.scatter(x['10'], x['20'], yn, color='black', marker='o') #Отмечаю точку с заданными координатами
    axes_2.text(x['10'], x['20'], yn, "y = f(x\u2081\u2080, x\u2082\u2080)") #Подписываю точку с заданными координатами

    x1, y = graph_2d(x1min, x1max, x20 = x['20'])

    axes_3 = fig.add_subplot(2, 2, 3) #Задаю позицию графика в едином окне
    axes_3.set(xlabel = ("x\u2081"), ylabel = ("y = f(x\u2081)")) #Задаю названия осей
    axes_3.plot(x1, y, color = "#735184")
    axes_3.scatter(x['10'], yn, color='black', marker='o') #Отмечаю точку с заданными координатами
    axes_3.text(x['10'], yn, "y = f(x\u2081\u2080)") #Подписываю точку с заданными координатами

    x2, y = graph_2d(x2min, x2max, x10 = x['10'])

    axes_4 = fig.add_subplot(2, 2, 4) #Задаю позицию графика в едином окне
    axes_4.set(xlabel = ("x\u2082"), ylabel = ("y = f(x\u2082)")) #Задаю названия осей
    axes_4.plot(x2, y, color = "#735184")
    axes_4.scatter(x['20'], yn, color='black', marker='o') #Отмечаю точку с заданными координатами
    axes_4.text(x['20'], yn, "y = f(x\u2082\u2080)") #Подписываю точку с заданными координатами

    plt.show()

if __name__ == "__main__":
    x1min = -500.
    x1max = 500.
    x2min = -500.
    x2max = 500.
    x = {'10': 420.9687, '20': 420.9687}

    draw(x1min, x1max, x2min, x2max)
