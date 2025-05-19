import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt

#Вариант 13
def function(x, A = 10): #Функция, вычисляющее значение выражения из задания
    return ((np.sin(A * np.pi * x)) / (2 * x)) + (x - 1)**4

def make_path(path): #Функция для проверки существования и создания папки
    if path.is_dir() == False: #Проверяю является ли указанный элемент папкой (если папки нет возвращает False)
        Path.mkdir("results") #Создаю заданную папку

def write_file(x, y, count): #Функция для записи в файл
    with open(Path("results", "output_ex1.csv"), mode='w', newline='') as file:
        fieldnames = ['No', 'x', 'y'] #Обозначаю заглавия столбцов
        writer = csv.DictWriter(file, fieldnames=fieldnames) 
        writer.writeheader()

        for n in range(count):
            line = {'No': str(n), 'x': str(x[n]), 'y': str(y[n])} #Задаю как словарь строк, т.к. writerow просит ОДНУ string-переменную
            writer.writerow(line)  

def draw(x, y, A = 10, xmin = -5.12, xmax = 5.12): #Функция для вывода графика
    plt.plot(x, y) #Вывожу точки с координатами x и y
    plt.title(f"Зависимость y от x при А = {A} и x ∈ [{xmin};{xmax}]")
    plt.ylabel("y = f(x)")
    plt.xlabel("x")
    plt.grid(True) #Включаю показ сетки на графике
    plt.show()

if __name__ == "__main__":    
    xmin = -5.12
    xmax = 5.12
    count = 256

    xdata = np.linspace(xmin, xmax, count) #Задаю множество x от xmin до xmax с шагом, соответствующим кол-ву count
    ydata = function(xdata) 

    make_path(Path("results"))

    write_file (xdata, ydata, count)

    draw(xdata, ydata)
