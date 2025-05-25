import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt

#Вариант 13
#Функция, вычисляющая значение выражения из задания
def function(x, A = 10):
    return ((np.sin(A * np.pi * x)) / (2 * x)) + (x - 1)**4

#Функция для проверки существования и создания папки
def make_path(path):
    #Проверяю является ли элемент папкой (если папки нет возвращает False)
    if path.is_dir() == False:
        #Создаю заданную папку
        Path.mkdir("results")

#Функция для записи в файл
def write_file(x, y, count):
    with open(Path("results", "output_ex1.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames = ['No', 'x', 'y']) 
        writer.writeheader()

        for n in range(count):
            #Задаю как словарь строк, т.к. writerow просит ОДНУ string-переменную
            line = {'No': str(n), 'x': str(x[n]), 'y': str(y[n])}
            writer.writerow(line)  

#Функция для вывода графика
def draw(x, y, A = 10, xmin = -5.12, xmax = 5.12):
    plt.plot(x, y)
    plt.title(f"Зависимость y от x при А = {A} и x ∈ [{xmin};{xmax}]")
    plt.ylabel("y = f(x)")
    plt.xlabel("x")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":    
    xmin = -5.12
    xmax = 5.12
    count = 256

    #Задаю множество x от xmin до xmax с шагом, соответствующим кол-ву count
    xdata = np.linspace(xmin, xmax, count)
    ydata = function(xdata) 

    make_path(Path("results"))

    write_file (xdata, ydata, count)

    draw(xdata, ydata)
