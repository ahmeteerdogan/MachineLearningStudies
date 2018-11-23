'''
Ensemble Learning(Kollektif Öğrenme)
Birden fazla sınıflandırma algoritması veya birden fazla tahmin algoritması
aynı anda kullanılarak daha başarılı bir sonuç çıkartılabilir.
Hem sınıflandırma hemde tahmin algoritmalarında kullanılabilir.

Ensemble Learning Yönetimleri
1-Random Forest

Birden fazla Decision Tree'nin ayn problem için aynı veri kümesi üzerinde 
çizilmesi ve daha sonra problemin çözülmesinde hep birlikte kullanılmasına denir.
Amaç: Veri kümesini birden faazzla küçük parçaya bölüp her parçadan birden fazla karar ağacı oluşturmak
ve sonrasında da o karar ağaçlarının sonuçlarını birleştirmek.
Veri arttıkça başarı düşer gibi de bir bulgu vardır.

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

#RandomForestModel
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)#n_estimators kaç tane karar ağacı çizileceğini belirtir.
rf_reg.fit(x,y)

print(rf_reg.predict(6.5))

plt.scatter(x,y,color='red')
plt.plot(x,rf_reg.predict(x),color='blue')



































