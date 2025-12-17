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

    # Параметры моделирования
    # Заданная диэлектрическая проницаемость
    eps1 = 4.5

    # Минимальная заданная частота в Гц
    f_min = 0.1e9

    # Максимальная заданная частота в Гц
    f_max = 0.4e9

    # Число Куранта
    Sc = 1.0
    
    # Дискрет по пространству в м
    dx = 6e-4

    # Дискрет по времени в с
    dt = dx * Sc / c   

    # Время расчета в секундах
    maxTime_s = 10e-8

    # Размер области моделирования в м
    maxSize_m = 6.0

    # Положение источника в м
    sourcePos_m = 1.0

    # Координаты датчика в м
    probePos_m = 2.0

    # Время расчета в секундах
    maxTime = int((maxTime_s/dt))

    # Размер области моделирования в отсчетах
    maxSize = int(maxSize_m/dx)

    # Положение источника в отсчетах
    sourcePos = int(sourcePos_m/dx)

    # Координаты датчика в отсчетах
    probePos = int(probePos_m/dx)
    
    # Где начинается поглощающий диэлектрик
    layer_loss_x = maxSize - 30

    # Моделирование
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[:] = eps1

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    # Потери в среде. loss = sigma * dt / (2 * eps * eps0)
    loss = np.zeros(maxSize)
    loss[layer_loss_x:] = 0.02

    # Магнитные потери в среде. loss_m = sigma_m * dt / (2 * mu * mu0)
    loss_m = np.zeros(maxSize - 1)
    loss_m[:] = loss[:-1]

    # Коэффициенты для расчета поля E
    ceze = (1.0 - loss) / (1.0 + loss)
    cezh = (Sc * Z0) / (eps * (1.0 + loss))

    # Коэффициенты для расчета поля H
    chyh = (1.0 - loss_m) / (1.0 + loss_m)
    chye = Sc / (mu * Z0 * (1.0 + loss_m))

    # Усреднение коэффициентов на границе поглощающего слоя
    ceze[layer_loss_x] = (ceze[layer_loss_x - 1] + ceze[layer_loss_x + 1]) / 2
    cezh[layer_loss_x] = (cezh[layer_loss_x - 1] + cezh[layer_loss_x + 1]) / 2

    Ez = np.zeros(maxSize)
    Ez_spectrum = np.zeros(maxTime)
    Hy = np.zeros(maxSize - 1)

    # Поле, зарегистрированное в датчике в зависимости от времени
    probeTimeEz = np.zeros(maxTime)

    source = GaussianDiff(magnitude=1, a_max=100, f_max=f_max, dt=dt, eps=eps[sourcePos], mu=mu[sourcePos])

    # Создание экземпляра класса для отображения распределения поля в пространстве
    display = AnimateFieldDisplay(maxSize, -1.1, 1.1, 'Ez, В/м', dx)
    display.activate()
    display.drawSources([sourcePos_m])
    display.drawProbes([probePos_m])
    display.drawBoundary([layer_loss_x * dx])

    for q in range(1, maxTime):
        # Расчет компоненты поля H        
        Hy[:] = chyh[:] * Hy[:] + chye[:] * (Ez[1:] - Ez[:-1])
        Hy[0] = 0

        # Источник возбуждения с использованием метода TFSF
        Hy[sourcePos - 1] -= (Sc / (Z0 * mu[sourcePos - 1])) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = ceze[1:-1] * Ez[1:-1] + cezh[1:-1] * (Hy[1:] - Hy[:-1])

        # Источник возбуждения с использованием метода TFSF
        Ez[sourcePos] += (Sc / np.sqrt(eps[sourcePos] * mu[sourcePos])) *  source.getE(-0.5, q + 0.5)

        probeTimeEz[q] = Ez[probePos]

        # Модификатор точности анимации распространения волны
        if q % 300 == 0:
            display.updateData(Ez, q * dt)

    display.stop()

    # Построение графика распространения волны
    time = np.arange(maxTime)
    fig = plt.figure(figsize = [16, 9])
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set(xlabel = 'Время t, с', ylabel = 'Ez, В/м')
    ax1.set_xlim(0, maxTime_s)
    ax1.set_ylim(-1.1, 1.1)
    ax1.plot(time * dt, probeTimeEz)
    ax1.grid()

    # Расчёт дискрета по частоте
    df = 1 / maxTime_s

    spectrum = np.abs(np.fft.fft(probeTimeEz))
    spectrum = np.fft.fftshift(spectrum)

    # Построение графика амплитудного спектра
    freq = np.arange(-(maxTime / 2) * df, (maxTime / 2) * df, df)
    ax2 = fig.add_subplot(1, 2, 2)   
    ax2.set(xlabel = 'Частота f, Гц', ylabel = '|S| / |Smax|')
    ax2.set_xlim(0, f_max)
    ax2.plot(freq, spectrum / np.max(spectrum))
    ax2.grid()

    plt.show()
