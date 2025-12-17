from typing import List
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt 

class AnimateFieldDisplay:
    # Класс для отображения анимации распространения ЭМ волны в пространстве
    def __init__(self, maxXSize: int, minYSize: float, maxYSize: float, yLabel: str, dx: float):
        self._maxXSize = maxXSize
        self._minYSize = minYSize
        self._maxYSize = maxYSize
        self._xdata = None
        self._line = None
        self._xlabel = 'Координата x, м'
        self._ylabel = yLabel
        self._dx = dx
        self._dt = dt
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'

    def activate(self):
        self._xdata = np.arange(self._maxXSize)

        # Включает интерактивный режим для анимации
        plt.ion()

        # Создаёт окна для графика
        self._fig, self._ax = plt.subplots()

        # Ставит отображаемые интервалы по осям
        self._ax.set_xlim(0, self._maxXSize * self._dx)
        self._ax.set_ylim(self._minYSize, self._maxYSize)

        # Ставит метки по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включает сетку на графике
        self._ax.grid()

        # Отображает поле в начальный момент времени
        self._line = self._ax.plot(self._xdata * self._dx, np.zeros(self._maxXSize))[0]

    def drawProbes(self, probesPos: List[float]):
        # Отображает положение датчиков
        self._ax.plot(probesPos, [0] * len(probesPos), self._probeStyle)

    def drawSources(self, sourcesPos: List[float]):
        # Отображает положение источников
        self._ax.plot(sourcesPos, [0] * len(sourcesPos), self._sourceStyle)

    def drawBoundary(self, position: float):
        # Рисует границу в области моделирования.
        self._ax.plot([position, position], [self._minYSize, self._maxYSize], '--k')

    def stop(self):
        plt.ioff()

    def updateData(self, data: npt.NDArray, timeCount: int):
        self._line.set_ydata(data)
        self._ax.set_title(str(timeCount) + " c")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

class GaussianDiff():
    # Источник, создающий дифференцированный гауссов импульс, распространяющийся только вправо
    def __init__(self, magnitude, a_max, f_max, dt, Sc: float = 1.0, eps: float = 1.0, mu: float = 1.0):
        # magnitude - максимальное значение в источнике;
        # dg - коэффициент, задающий начальную задержку гауссова импульса;
        # wg - коэффициент, задающий ширину гауссова импульса.
        self.magnitude = magnitude
        self.a_max = a_max
        self.f_max = f_max
        self.dt = dt
        self.Sc = Sc
        self.eps = eps
        self.mu = mu    

    def getE(self, pos, time):
        wg = np.sqrt(np.log(5.5 * self.a_max)) / (np.pi * self.f_max)
        dg = wg * np.sqrt(np.log(2.5 * self.a_max * np.sqrt(np.log(2.5 * self.a_max))))
        e = ((time - pos * np.sqrt(self.eps * self.mu) / self.Sc) - dg/self.dt) / (wg/self.dt)
        return (-2 * self.magnitude * e * np.exp(-(e ** 2)))
    
class Probe:
    # Класс для хранения временного сигнала в датчике
    def __init__(self, position: int, maxTime: int):
        # position - положение датчика (номер ячейки)
        # maxTime - максимальное количество временных шагов для хранения в датчике
        self.position = position

        # Временные сигналы для полей E и H
        self.E = np.zeros(maxTime)
        self.H = np.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E: npt.NDArray, H: npt.NDArray):
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1
    
if __name__ == '__main__':
    # Используемые константы
    # Характеристическое сопротивление свободного пространства
    Z0 = 120.0 * np.pi

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    # Магнитная постоянная
    mu0 = np.pi * 4e-7

    # Скорость света в вакууме
    c = 1.0 / np.sqrt(mu0 * eps0)

    # Минимальная заданная частота в Гц
    f_min = 0

    # Максимальная заданная частота в Гц
    f_max = 1.5e9

    # Число Куранта
    Sc = 1.0
    
    # Дискрет по пространству в м
    dx = 6e-4

    # Дискрет по времени в с
    dt = dx * Sc / c   

    # Время расчета в секундах
    maxTime_s = 6e-8

    # Размер области моделирования в м
    maxSize_m = 6.0
    layer4_m = maxSize_m - 1
    layer3_m = layer4_m - 0.15
    layer2_m = layer3_m - 0.09
    layer1_m = layer2_m - 0.13

    # Положение источника в м
    sourcePos_m = 1.0

    # Время расчета в секундах
    maxTime = int((maxTime_s/dt))

    # Размер области моделирования в отсчетах
    maxSize = int(maxSize_m/dx)
    layers = [int(layer4_m/dx), int(layer3_m/dx), int(layer2_m/dx), int(layer1_m/dx)]

    # Положение источника в отсчетах
    sourcePos = int(sourcePos_m/dx)

    # Координаты датчика в отсчетах
    probesPos = [int((sourcePos_m - 0.5)/dx), int((sourcePos_m + 0.5)/dx)]
    probes = [Probe(pos, maxTime) for pos in probesPos]
    
    # Где начинается поглощающий диэлектрик
    # layer_loss_x = maxSize - 25

    # Моделирование
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[:] = 1
    eps[layers[0]:] = 7.2
    eps[layers[1]:] = 8.5
    eps[layers[2]:] = 12.3
    eps[layers[3]:] = 6.3

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    # Коэффициенты для расчета ABC второй степени
    # Sc для правой границы
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Sc для левой границы
    Sc1Left = Sc / np.sqrt(mu[0] * eps[0])

    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)

    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = np.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = np.zeros(3)

    source = GaussianDiff(magnitude=1, a_max=100, f_max=f_max, dt=dt, eps=eps[sourcePos], mu=mu[sourcePos])

    # Создание экземпляра класса для отображения распределения поля в пространстве
    display = AnimateFieldDisplay(maxSize, -1.1, 1.1, 'Ez, В/м', dx)
    display.activate()
    display.drawSources([sourcePos_m])
    for pos in probesPos:
        display.drawProbes([pos * dx])
    for layer in layers:
        display.drawBoundary([layer * dx])

    for q in range(1, maxTime):
        # Расчет компоненты поля H        
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)

        # Источник возбуждения с использованием метода TFSF
        Hy[sourcePos - 1] -= (Sc / (Z0 * mu[sourcePos - 1])) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * Z0 / eps[1:-1]

        # Источник возбуждения с использованием метода TFSF
        Ez[sourcePos] += (Sc / np.sqrt(eps[sourcePos] * mu[sourcePos])) *  source.getE(-0.5, q + 0.5)

        # Граничные условия ABC второй степени (слева)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]

        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        # Модификатор точности анимации распространения волны
        if q % 300 == 0:
            display.updateData(Ez, q * dt)

    display.stop()

    # Построение графика распространения волны
    time = np.arange(maxTime)
    fig = plt.figure(figsize = [16, 9])
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set(xlabel = 'Время t, с', ylabel = 'Ez, В/м')
    ax1.set_xlim(0, maxTime_s)
    ax1.set_ylim(-1.1, 1.1)
    ax1.plot(time * dt, probes[0].E)
    ax1.plot(time * dt, probes[1].E)
    ax1.grid()

    # Расчёт дискрета по частоте
    df = 1 / maxTime_s
    
    # Расчёт спектра падающей волны
    spectrum = np.abs(np.fft.fft(probes[0].E))
    spectrum = np.fft.fftshift(spectrum)

    # Построение графика амплитудного спектра падающей волны
    freq = np.arange(-(maxTime / 2) * df, (maxTime / 2) * df, df)
    ax2 = fig.add_subplot(2, 2, 2)   
    ax2.set(xlabel = 'Частота f, Гц', ylabel = '|S| / |Smax|')
    ax2.set_xlim(0, f_max)
    ax2.plot(freq, spectrum / np.max(spectrum))
    ax2.grid()

    # Расчёт спектра отраженной волны
    spectrum = np.abs(np.fft.fft(probes[1].E))
    spectrum = np.fft.fftshift(spectrum)

    # Построение графика амплитудного спектра отраженной волны
    ax2.plot(freq, spectrum / np.max(spectrum))

    koef_otr = np.abs(np.fft.fft(probes[0].E)/np.fft.fft(probes[1].E))
    koef_otr = np.fft.fftshift(koef_otr)

    ax3 = fig.add_subplot(2, 2, 3)   
    ax3.set(xlabel = 'Частота f, Гц', ylabel = 'Коэффициент отражения Г')
    ax3.set_xlim(f_min, f_max)
    ax3.set_ylim(0, 1)
    ax3.plot(freq, koef_otr)
    ax3.grid()

    plt.show()