"""
@author: ahmeterdogan
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('satislar.csv')

#2.2 Veri on isleme (Ayrı ayrı kolonlarımı ele aldım)
aylar = veriler[['Aylar']]
print(aylar)
satislar = veriler[['Satislar']]
print(satislar)

#verilerin egitim ve test icin bolunmesi  (x_train'deki aylara ait y_train de
# bulunan satışlara bakarak eğitim gerçekleşecek.Ardından da x_test'e göre bir tahmin elde edecek
#ardından karşılaştırma yapılacak)
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
'''
--Verileri belli bir standarda indirgiyoruz--

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

'''
#model inşası (linear regression)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
#eğitim
lr.fit(x_train,y_train)
#tahmin et
tahmin = lr.predict(x_test)
'''X_Trainden yola çıkarak Y_Traini tahmin edecek.''' 

#Veri Görselleştirme
 #Eğer verileri sıralamazsam random olarak çizmeye çalışacak eğer index'ine göre sıralarsam
 #Doğru çizebilirim
x_train=x_train.sort_index()
y_train=y_train.sort_index()
 
plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)




    
    

