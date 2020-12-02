# -*- coding: utf-8 -*-
# Дз_02.12_2020

import pandas as pd               # Импортируем библиотеку pandas

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt   # Импортируем библиотеку matplotlib.pyplot
# %matplotlib inline

data = pd.read_excel('https://github.com/anastasiarazb/skillbox_nlp_demo/blob/master/IPG2211A2N.xls?raw=true',
                     skiprows=10) # Считываем эксель файл с данными

dates = pd.to_datetime(data['observation_date'],dayfirst=True) #dayfirst считываем день первым числом

# переворачиваем данные от новых к старым
data['dates'] = dates             # создаем новый столбец с именем dates с данными
data = data.sort_values('dates')  # сортировка данных
data = data.reset_index()         # исправляем номерацию строк

x = data['dates']                 # Присваиваем значению х массив с двнными dates
y = data['IPG2211A2N']            # Присваиваем значению y массив с двнными IPG2211A2N
plt.figure(figsize=(20, 5))       # Указание размера графика
plt.plot(x, y)                    # Построение графика

plt.xlabel('observation_date')    # Указание названия оси Х графика
plt.ylabel('IPG')                 # Указание названия оси Y графика

# Так как на разных участках графика данные имеют разный расброс, то выбираем несколько диапазона данных
plt.figure(figsize=(20, 5)) # Указание размера графика
# 1 - диапазон 1939 - 1947 гг.
data_short1 = data[data['dates'].between(pd.Timestamp(day=1, month=1, year=1939), pd.Timestamp(day=1, month=1, year=1947))]
plt.plot(data_short1['dates'], data_short1['IPG2211A2N'], label='IPG1')
plt.legend()                # Указание вывести на график легенду(обозначение)
# 2 - диапазон 1947 - 1972 гг.
data_short2 = data[data['dates'].between(pd.Timestamp(day=1, month=1, year=1947), pd.Timestamp(day=1, month=1, year=1972))]
plt.plot(data_short2['dates'], data_short2['IPG2211A2N'], label='IPG2')
plt.legend()                # Указание вывести на график легенду(обозначение)
# 3 - диапазон 1972 - 2020 гг.
data_short3 = data[data['dates'].between(pd.Timestamp(day=1, month=1, year=1972), pd.Timestamp(day=1, month=1, year=2020))]
plt.plot(data_short3['dates'], data_short3['IPG2211A2N'], label='IPG3')
plt.legend()                # Указание вывести на график легенду(обозначение)

plt.xlabel('observation_date')
plt.ylabel('IPG')

plt.figure(figsize=(4, 5)) # Указание размера графика
data_short1 = data[data['dates'].between(pd.Timestamp(day=1, month=1, year=1939), pd.Timestamp(day=1, month=1, year=1947))]
plt.plot(data_short1['dates'], data_short1['IPG2211A2N'])

plt.xlabel('observation_date')
plt.ylabel('IPG1')

plt.figure(figsize=(6, 5)) # Указание размера графика
data_short2 = data[data['dates'].between(pd.Timestamp(day=1, month=1, year=1947), pd.Timestamp(day=1, month=1, year=1972))]
plt.plot(data_short2['dates'], data_short2['IPG2211A2N'], color='orange')

plt.xlabel('observation_date')
plt.ylabel('IPG2')

plt.figure(figsize=(12, 5)) # Указание размера графика
data_short3 = data[data['dates'].between(pd.Timestamp(day=1, month=1, year=1972), pd.Timestamp(day=1, month=1, year=2020))]
plt.plot(data_short3['dates'], data_short3['IPG2211A2N'], color='green')


plt.xlabel('observation_date')
plt.ylabel('IPG3')

# Скользящее среднее сглаживание
plt.figure(figsize=(20, 5)) # Указание размера графика
x = data['dates']

data['rolling_mean_5'] = data_short1['IPG2211A2N'].rolling(5).mean()
plt.plot(data_short1['dates'], data_short1['IPG2211A2N'], label='IPG1')
plt.plot(x, data['rolling_mean_5'], label='rolling_mean_5', color='red')

data['rolling_mean_10'] = data_short2['IPG2211A2N'].rolling(10).mean()
plt.plot(data_short2['dates'], data_short2['IPG2211A2N'], label='IPG2')
plt.plot(x, data['rolling_mean_10'], label='rolling_mean_10', color='blue')

data['rolling_mean_30'] = data_short3['IPG2211A2N'].rolling(30).mean()
plt.plot(data_short3['dates'], data_short3['IPG2211A2N'], label='IPG3')
plt.plot(x, data['rolling_mean_30'], label='rolling_mean_30', color='yellow')

plt.legend()                    # Указание вывести на график легенду(обозначение)
plt.xlabel('observation_date')  # Указание названия оси X графика
plt.ylabel('IPG')               # Указание названия оси Y графика

data['IPG2211A2N'].hist(bins=18)        #Строим гистограмму для всего объема данных bins - уровень детализации
data_short1['IPG2211A2N'].hist(bins=1)  #Строим гистограмму для данных участка 1
data_short2['IPG2211A2N'].hist(bins=8)  #Строим гистограмму для данных участка 2
data_short3['IPG2211A2N'].hist(bins=18) #Строим гистограмму для данных участка 3
plt.xlabel('frequency of occurrence')
plt.ylabel('IPG')

pd.options.mode.chained_assignment = None  # default='warn'отключить предупреждение
plt.figure(figsize=(10, 5))         # Указание размера графика
x = data_short1['dates']            # Присваиваем значению х массив с двнными dates
y = data_short1['IPG2211A2N']       # Присваиваем значению y массив с двнными IPG2211A2N
data_short1['quantile_95'] = data_short1['IPG2211A2N'].quantile(0.95)
data_short1['quantile_05'] = data_short1['IPG2211A2N'].quantile(0.05)
plt.plot(x, y, label='IPG1')
plt.plot(x, data_short1['quantile_95'], label='quantile_95')
plt.plot(x, data_short1['quantile_05'], label='quantile_05')

plt.legend()                       # Указание вывести на график легенду(обозначение)
plt.xlabel('observation_date')     # Указание названия оси Х графика
plt.ylabel('IPG1')                 # Указание названия оси Y графика

plt.figure(figsize=(10, 5))         # Указание размера графика
x = data_short2['dates']            # Присваиваем значению х массив с двнными dates
y = data_short2['IPG2211A2N']       # Присваиваем значению y массив с двнными IPG2211A2N
data_short2['quantile_95'] = data_short2['IPG2211A2N'].quantile(0.95)
data_short2['quantile_05'] = data_short2['IPG2211A2N'].quantile(0.05)
plt.plot(x, y, label='IPG2')
plt.plot(x, data_short2['quantile_95'], label='quantile_95')
plt.plot(x, data_short2['quantile_05'], label='quantile_05')

plt.legend()                       # Указание вывести на график легенду(обозначение)
plt.xlabel('observation_date')     # Указание названия оси Х графика
plt.ylabel('IPG2')                 # Указание названия оси Y графика

plt.figure(figsize=(10, 5))        # Указание размера графика
x = data_short3['dates']           # Присваиваем значению х массив с двнными dates
y = data_short3['IPG2211A2N']      # Присваиваем значению y массив с двнными IPG2211A2N
data_short3['quantile_95'] = data_short3['IPG2211A2N'].quantile(0.95)
data_short3['quantile_05'] = data_short3['IPG2211A2N'].quantile(0.05)
plt.plot(x, y, label='IPG3')
plt.plot(x, data_short3['quantile_95'], label='quantile_95')
plt.plot(x, data_short3['quantile_05'], label='quantile_05')

plt.legend()                       # Указание вывести на график легенду(обозначение)
plt.xlabel('observation_date')     # Указание названия оси Х графика
plt.ylabel('IPG3')                 # Указание названия оси Y графика



plt.figure(figsize=(10, 5))                                 # Указание размера графика
pd.plotting.autocorrelation_plot(data_short1['IPG2211A2N']) # Построение графика авторорреляции для участка 1
plt.locator_params(axis='x', nbins=10)                      # Указание размера сетки на графике

plt.figure(figsize=(15, 5))                                 # Указание размера графика
pd.plotting.autocorrelation_plot(data_short2['IPG2211A2N']) # Построение графика авторорреляции для участка 2
plt.locator_params(axis='x', nbins=30)                      # Указание размера сетки на графике

plt.figure(figsize=(20, 5))                                 # Указание размера графика
pd.plotting.autocorrelation_plot(data_short3['IPG2211A2N']) # Построение графика авторорреляции для участка 3
plt.locator_params(axis='x', nbins=60)                      # Указание размера сетки на графике

# Постороение Трендов

from sklearn.linear_model import LinearRegression # из библиотеки sklearn.linear_model вызываем ф-ю LinearRegression

X1 = pd.DataFrame(data_short1.index)  # обучащие данные 1, таблица из одной колонки с номерами строк
X2 = pd.DataFrame(data_short2.index)  # обучащие данные 2
X3 = pd.DataFrame(data_short3.index)  # обучащие данные 3
Y1 = data_short1['IPG2211A2N']  # "правильные ответы" 1
Y2 = data_short2['IPG2211A2N']  # "правильные ответы" 2
Y3 = data_short3['IPG2211A2N']  # "правильные ответы" 3

model1 = LinearRegression().fit(X1, Y1) # cоздаем модель 1, для обучения
model2 = LinearRegression().fit(X2, Y2) # cоздаем модель 2, для обучения
model3 = LinearRegression().fit(X3, Y3) # cоздаем модель 3, для обучения

trend1 = model1.predict(X1) # отрисовка линии тренда для участка 1
trend2 = model2.predict(X2) # отрисовка линии тренда для участка 1
trend3 = model3.predict(X3) # отрисовка линии тренда для участка 1

# Построение графиков с линиями тренда
plt.figure(figsize=(20, 8))                           # Указание размера графика
x1 = data_short1['dates']                             # Присваиваем значению х1 массив с двнными dates
plt.plot(x1, data_short1['IPG2211A2N'], label='IPG1') # Построение графика данных участка 1
plt.plot(x1, trend1, label='trend1')                  # Построение графика линии тренда участка 1

x2 = data_short2['dates']                             # Присваиваем значению х2 массив с двнными dates
plt.plot(x2, data_short2['IPG2211A2N'], label='IPG2') # Построение графика данных участка 2
plt.plot(x2, trend2, label='trend2')                  # Построение графика линии тренда участка 2

x3 = data_short3['dates']                             # Присваиваем значению х3 массив с двнными dates
plt.plot(x3, data_short3['IPG2211A2N'], label='IPG3') # Построение графика данных участка 3
plt.plot(x3, trend3, label='trend3')                  # Построение графика линии тренда участка 3

plt.legend()                                          # Указание вывести на график легенду(обозначение)
plt.xlabel('observation_date')                        # Указание названия оси Х графика
plt.ylabel('IPG')                                     # Указание названия оси Y графика
