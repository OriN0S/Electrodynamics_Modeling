import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from urllib import request
from pathlib import Path
import csv
import json

#Вариант 13
class RCS:
    def __init__(self, fmin, fmax, diameter):
        self.diameter = diameter
        self.fmin = fmin
        self.fmax = fmax
        self.radius = diameter / 2

    def hn_function(self, n, x):
        return sp.spherical_jn(n, x) + (sp.spherical_yn(n, x) * 1j) #Возвращаю комплексное число (действительная часть слева, комплексная - справа)
 
    def an_function(self, n, k):
        return (sp.spherical_jn(n, k * self.radius)) / (self.hn_function(n, k * self.radius))
        
    def bn_function(self, n, k):
        numerator = (k * self.radius * sp.spherical_jn(n - 1, k * self.radius)) - (n * sp.spherical_jn(n, k * self.radius))
        denominator = (k * self.radius * self.hn_function(n - 1, k * self.radius)) - (n * self.hn_function(n, k * self.radius))
        return (numerator / denominator)

    def calculate(self, count_p, count_n = 50):
        sum = 0
        frequencies = np.linspace(fmin, fmax, count_p)
        wavelengths = (3 * (10**8)) / frequencies     
        k = (2 * np.pi) / wavelengths
        for n in range(count_n):
            sum += ((-1)**(n + 1)) * ((n + 1) + 0.5) * (self.bn_function((n + 1), k) - self.an_function((n + 1), k)) 
        rcs = (wavelengths**2 / np.pi) * (np.abs(sum)**2)
        return frequencies, wavelengths, rcs
        
class Output:
    def __init__(self, frequencies, wavelengths, rcs):
        self.freq = frequencies
        self.wave = wavelengths
        self.rcs = rcs

    def write_file(self, count):
        data = []
        for n in range(count):
            data += [{"freq": self.freq[n], "lambda": self.wave[n], "rcs": self.rcs[n]}]

        result = {"data": data}

        self.make_path(Path("results"))

        with open(Path("results", "output_ex4.json"), mode='w') as file:
            file.write(json.dumps(result, indent = 4, separators = (", ", ": ")))

    def draw(self):
        plt.plot(self.freq, self.rcs) #Вывожу точки с координатами x и y
        plt.title("Зависимость ЭПР идеально проводящей сферы \u03c3 от частоты F")
        plt.ylabel("\u03c3 = f(F)")
        plt.xlabel("F")
        plt.grid(True) #Включаю показ сетки на графике
        plt.show()

    def make_path(self, path): #Функция для проверки существования и создания папки
        if path.is_dir() == False: #Проверяю является ли указанный элемент папкой (если папки нет возвращает False)
            Path.mkdir("results") #Создаю заданную папку
    
def load_file(url): #Функция для скачивания файла с сайта и его прочтения
        file_path = Path("results", "task_rcs_01.csv")
        request.urlretrieve(url, file_path)

        with open (file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file, fieldnames = ['Вариант', 'fmin', 'fmax', 'D'])
            for row in reader:
                if row['Вариант'] == '13':
                    fmin = float(row['fmin'])
                    fmax = float(row['fmax'])
                    diameter = float(row['D'])

        return fmin, fmax, diameter

if __name__ == "__main__":
    fmin, fmax, diameter = load_file("https://jenyay.net/uploads/Student/Modelling/task_rcs_01.csv")
    count = 1000
    
    find_rcs = RCS(fmin, fmax, diameter)
    frequencies, wavelengths, rcs = find_rcs.calculate(count)
    out = Output(frequencies, wavelengths, rcs)
    out.write_file(count)
    out.draw()