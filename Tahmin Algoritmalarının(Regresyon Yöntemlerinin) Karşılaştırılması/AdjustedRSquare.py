# -*- coding: utf-8 -*-
'''
###Adjusted R²###
Multi Linear Regression kullanılan bir veri kümesine 
yeni bir değişken eklendiğinde ve eklenen değişken sisteme 
olumsuz olması durumunda için R² değeri azalmıyor hatta hiç etkilenmiyor.
Yalnızca değişken sisteme olumlu  etki sağladığında R² artıyor. Aksi taktirde azalma söz konu bile olmuyor.
Şayet eklenen olumsuz bir değişken sistemi olumsuz olarak etkilerken R² başarılı bir görüş almamızı engellemiş oluyor.

Adjucted R²=1 - ( 1 - R² )*((n - 1)/(n - p - 1))

n=kaç tane elemanı olduğunu tutar.
p=kaç tane değişken aldığımızı tutar.
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

#RandomForestModeline ait R2 hesaplaması
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)#n_estimators kaç tane karar ağacı çizileceğini belirtir.
rf_reg.fit(x,y)

print(rf_reg.predict(6.5))

plt.scatter(x,y,color='red')
plt.plot(x,rf_reg.predict(x),color='blue')
plt.show() 


#R² Hesaplama
from sklearn.metrics import r2_score
print('Random Forest R2 degeri:')
print(r2_score(y,rf_reg.predict(x))) #y burada benim bağımlı değişkenim yani maaş      
                                     #bağımsız x'e göre de tahmin ettiği sonuçlarla birlikte fonksiyona sokuyoruz.





