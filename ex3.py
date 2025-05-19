import numpy as np
import matplotlib.pyplot as plt

#Вариант 13
def function(x, D = 2): #Функция, вычисляющее значение выражения из задания
    y2 = 0
    for i in range(D):
        y2 += x[i] * np.sin(np.sqrt(np.fabs(x[i])))
    y = 418.9829 * D + y2
    return y

def graph_2d(xmin, xmax, x10, x20, count = 200): #Функция для рассчёта выражения от одной переменной
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
    

def draw_2d(xn, y, n = None): #Функция для построения двухмерного графика
    plt.plot(xn, y) #Вывожу точки с координатами x и y

    match n: 
        case 1:
            plt.ylabel("y = f(x\u2081)")
            plt.xlabel("x\u2081")
        
        case 2:
            plt.ylabel("y = f(x\u2082)")
            plt.xlabel("x\u2082")

        case _:
            plt.ylabel("y = f(x)")
            plt.xlabel("x")

    plt.title(f"f(x\u2081\u2080, x\u2082\u2080) = {function([x['10'], x['20']])}") #Как заглавие вывожу значение выражения в x10 и x20
    plt.grid(True) #Включаю показ сетки на графике
    plt.show()

def draw_3d(x1, x2, z, elevation = 30, azimuth = -60): #Функция для построения трехмерного графика
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.set(xlabel="x\u2081", ylabel="x\u2082", zlabel="z = f(x\u2081, x\u2082)") #Задаю названия осей
    axes.view_init(elev=elevation, azim=azimuth) #Задаю точку наблюдения
    axes.plot_surface(x1, x2, z, cmap="hot")

    plt.title(f"f(x\u2081\u2080, x\u2082\u2080) = {function([x['10'], x['20']])}") #Как заглавие вывожу значение выражения в x10 и x20
    plt.show()

if __name__ == "__main__":
    x1min = -500.
    x1max = 500.
    x2min = -500.
    x2max = 500.
    x = {'10': 420.9687, '20': 420.9687}
    
    x1, x2, y = graph_3d(x1min, x1max, x2min, x2max)
    draw_3d(x1, x2, y)
    draw_3d(x1, x2, y, 90, 90)

    x1, y = graph_2d(x1min, x1max, None, x['20'])
    draw_2d(x1, y, 1)

    x2, y = graph_2d(x2min, x2max, x['10'], None)
    draw_2d(x2, y, 2)
