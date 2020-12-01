# -*- coding: utf-8 -*-
#pip install matplotlib
#pip install pandas

import matplotlib.pyplot as plt
%matplotlib inline

x = [1, 2, 6]
y = [5, 6, 7]
plt.figure(figsize=(15, 8))
plt.plot(x, y, label='y = f(x)', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')

import pandas as pd

data = pd.read_excel('https://github.com/anastasiarazb/skillbox_nlp_demo/blob/master/%D0%9C%D1%81%D0%BA_5%D0%BB%D0%B5%D1%82.xls?raw=true',
                     skiprows=6)

data.columns # вывод списка всех колонок документа

data['Местное время в Москве (ВДНХ)'][0] # вывод первого значения конкретной колонки

x = data.index
y = data['T']
plt.figure(figsize=(15, 8))
plt.plot(x, y)
plt.legend()
plt.xlabel('t')
plt.ylabel('T')

dates = pd.to_datetime(data['Местное время в Москве (ВДНХ)'],dayfirst=True) #dayfirst считываем день первым числом

# переворачиваем данные от новых к старым
# сортировка данных
data['dates'] = dates # создаем новый столбец с именем dates с данными
data = data.sort_values('dates')
# исправляем номерацию строк
data = data.reset_index()

x = data['dates']
y = data['T']
plt.figure(figsize=(15, 8))
plt.plot(x, y)
plt.legend()
plt.xlabel('t')
plt.ylabel('T')

# выбор диапозона по условию
start = pd.Timestamp(day=1, month=7, year=2019)
conditions = data['dates'] > start
data[conditions]

data_short = data[data['dates'] > pd.Timestamp(day=1, month=7, year=2019)]

# 1 способ выезки диапвзона данных
data_short1 = data[data['dates'] > pd.Timestamp(day=10, month=7, year=2019)]
data_short2 = data_short1[data['dates'] < pd.Timestamp(day=10, month=7, year=2020)]
plt.plot(data_short2['dates'], data_short2['T'])

# 2 способ выезки диапвзона данных
data_short2 = data[(data['dates'] > pd.Timestamp(day=10, month=7, year=2019)) 
                  & (data['dates'] < pd.Timestamp(day=10, month=7, year=2020))]
plt.plot(data_short2['dates'], data_short2['T'])

# 3 способ выезки диапвзона данных
data_short = data[data['dates'].between(pd.Timestamp(day=10, month=7, year=2019), pd.Timestamp(day=10, month=7, year=2020))]
plt.plot(data_short['dates'], data_short['T'])

x = data_short['dates']
y = data_short['T']
plt.figure(figsize=(20, 8))
plt.plot(x, y)
plt.legend()
plt.xlabel('t')
plt.ylabel('T')

data['T'].min()
data['T'].max()

# не оптимальный но короткий пример отрисовки значений экстремумов

data['T_min'] = data['T'].min()
data['T_max'] = data['T'].max()
data['T_mean'] = data['T'].mean()
data.head()

x = data['dates']
plt.plot(x, data['T'], label='T')
plt.plot(x, data['T_min'], label='T_min')
plt.plot(x, data['T_max'], label='T_max')
plt.plot(x, data['T_mean'], label='T_mean')
plt.legend()

# скользящее среднее сглаживание
data['T'].rolling(100).mean()

# вначале идут пустые значения, потомучто не достаточно данных для расчета значений

data['rolling_mean_100'] = data['T'].rolling(100, center=True).mean()
data['rolling_mean_500'] = data['T'].rolling(500, center=True).mean()
plt.figure(figsize=(20, 5))
x = data['dates']
plt.plot(x, data['T'], label='T')
plt.plot(x, data['T_min'], label='T_min')
plt.plot(x, data['T_max'], label='T_max')
plt.plot(x, data['T_mean'], label='T_mean')
plt.plot(x, data['rolling_mean_100'], label='rolling_mean_100', color='yellow')
plt.plot(x, data['rolling_mean_500'], label='rolling_mean_500', color='orange')
plt.legend()

#Гистограмма - графикб показыающий насколько часто встречается значение

data['T'].hist()
data['T'].hist(bins=100) #высокая детализация

# квантили - , например, 95% квантиль = 23.3, значит в 95% случаев значение не превышало 23.3

data['T'].quantile(0.95)
data['T'].quantile(0.05)

data['quantile_95'] = data['T'].quantile(0.95)
data['quantile_05'] = data['T'].quantile(0.05)

plt.figure(figsize=(20, 5))
x = data['dates']
plt.plot(x, data['T'], label='T')
plt.plot(x, data['T_min'], label='T_min')
plt.plot(x, data['T_max'], label='T_max')
plt.plot(x, data['T_mean'], label='T_mean')
plt.plot(x, data['quantile_95'], label='quantile_95', color='yellow')
plt.plot(x, data['quantile_05'], label='quantile_05', color='orange')
plt.legend()

# корреляция это величина показывающия на сколько похожи тренды двух графиковб если совпадат, то корреляция =1
data.corr()

# авторорреляция - ф-я показывает уровни корреляции исходного и сдвинутого на лаг рядов
pd.plotting.autocorrelation_plot(data['T'])

# Сейчас много точек в каждом дне года, нужно оставить только одно, чтобы лаг был равен 1 дню
data_daily = data[data['dates'].dt.hour == 12]

plt.figure(figsize=(20, 5))
pd.plotting.autocorrelation_plot(data_daily['T'])
plt.locator_params(axis='x', nbins=50)

# тренд

from sklearn.linear_model import LinearRegression

#на вход нужна двумерная таблица для этого создаем объект формата таблица pandas DataFrame

X = pd.DataFrame(data.index) # обучащие данные, таблица из одной колонки с номерами строк
Y = data['T'] # "правильные ответы" формат колонки

model = LinearRegression() # моздаем модель, для обучения
model.fit(X, Y) # обучаем модель

trend = model.predict(X) # отрисовать точки "идеальной" прямой для тех же позиций Х

x = data.index
plt.plot(x, data['T'])
plt.plot(x, trend)
