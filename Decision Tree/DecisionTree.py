#Decision Tree(Karar Ağacı)
'''
#Genelde sınıflandırma için kullanır fakat tahmin içinde kullanılabiliyor.
Verilerin dağılımı ile ilgili veriyi bir bölgeden ikiye ayırır.
'''


#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('maaslar.csv')
#pd.read_csv("veriler.csv")

#Data Frame
x = veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)
plt.scatter(x,y,color='red')
plt.plot(x,r_dt.predict(x),color='blue')

print(r_dt.predict(11))
print(r_dt.predict(6.6))


































