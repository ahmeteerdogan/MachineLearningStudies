# -*- coding: utf-8 -*-
"""
#R2 Hesaplanması

Yöntemleri Karşılaştırmak

ilk yapılacak işlem tahmin değerinin gerçek ile karşılaştırılması ile başlar.
adımlar
1-Formül=Hata Kareleri toplamı
(gerçekdeğer-tahmin)² 
Gerçekte olması gerekenle-Gerçekleşmeyen değer arasındaki farkın karesi
gerçek değer=10
tahmin=12
(10-12)'nin karesi alınır
kare alınmasının sebebi eksi(-) değer gelmesi durumundaki karışıklığı önlemek amacı ile 
mutlak değeri alınması daha mantıklı olduğu için.
hepsinin toplamı hesaplanır(her bir satır için)
2-Ortalama Farkların Toplamı
Daha sonra yine yine her satır için Gerçek değerden tahmin edilenlerin ortalaması çıkartılır çıkan sonucun yine karesi alınır.
(gerçekdeğer-tahminlerortalaması)² 
örneğin
gerçek değer=10
veri kümesinde tahmin edilen bütün değerlerin ortalaması=12
(10-12)² olarak hesaplanır 

Hepsinin toplamı hesaplanır(her bir satır için)



SON ADIM

R²=1-HKT(Hata Kareleri Toplamı)/Ortalama Farkların Toplamı


R² 0 ve 0'ın altına düşüyorsa algoritma çöp.
Hata karelerinin toplamı 0 olursa bu çok iyi demektir ki hiç hata yapılmamış
R²'nin 1 olma durumu algoritmanın ne ½100 çalışıyor olması demektir. Yani R² 
1'e ne kadar yakınsa algoritma o kadar iyi çalışıyor demektir.
"""
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
print(r2_score(y,rf_reg.predict(x)))
