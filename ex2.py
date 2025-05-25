from pathlib import Path
import csv
import matplotlib.pyplot as plt
import argparse

#Вариант 13
#Функция для считывания файла
def load_file(input):
        x = []
        y = []

        with open (Path("results", f"{input}"), mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                #Пополняю пустой список x значениями в столбце x в порядке вывода
                x += [float(row['x'])]
                #Пополняю пустой список y значениями в столбце y в порядке вывода
                y += [float(row['y'])]

        return x, y

#Функция для вывода графика
def draw(x, y, Legend = None):
    plt.plot(x, y)
    plt.ylabel("y = f(x)")
    plt.xlabel("x")
    plt.grid(True)
    if Legend != None:
        plt.legend([Legend])
    plt.show()

if __name__ == "__main__":
    #Задаю parser как хранилище аргументов
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str) 
    parser.add_argument("-l", "-leg", "--legend", type=str)
    #Считываю аргументы из хранилища parser и заношу их в arguments
    arguments = parser.parse_args()
    
    xdata, ydata = load_file(arguments.input_file)
       
    draw(xdata, ydata, arguments.legend)
