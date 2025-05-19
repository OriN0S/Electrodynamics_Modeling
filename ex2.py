from pathlib import Path
import csv
import matplotlib.pyplot as plt
import argparse

#Вариант 13
def load_file(input): #Функция для считывания файла
        x = [] #Задаю x как пустой список
        y = [] #Задаю y как пустой список

        with open (Path("results", f"{input}"), mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x += [float(row['x'])] #Пополняю пустой список x значениями в столбце x в порядке вывода
                y += [float(row['y'])] #Пополняю пустой список y значениями в столбце y в порядке вывода

        return x, y

def draw(x, y, Legend = None): #Функция для вывода графика
    plt.plot(x, y) #Вывожу точки с координатами x и y
    plt.ylabel("y = f(x)")
    plt.xlabel("x")
    plt.grid(True) #Включаю показ сетки на графике
    if Legend != None:
        plt.legend([Legend])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser() #Задаю parser как хранилище аргументов
    parser.add_argument('input_file', type=str) 
    parser.add_argument("-l", "-leg", "--legend", type=str)
    arguments = parser.parse_args() #Считываю аргументы из хранилища parser и заношу их в arguments

    xdata, ydata = load_file(arguments.input_file)
       
    draw(xdata, ydata, arguments.legend)
