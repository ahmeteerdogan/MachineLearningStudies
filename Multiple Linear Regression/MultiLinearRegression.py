#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")


#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(c)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)



#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


boy=s2.iloc[:,3:4].values
print(boy)
#Bütün satıları al 3. satıra kadar
sol=s2.iloc[:,:3]
#bütün satırları al ve 4.satırdan sonrakileri al
sag=s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)
r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred=r2.predict(x_test)

#P-değeri hesaplama
#Backward Elemination kullanımı

import statsmodels.formula.api as sm
#Dizi oluşturup bütün değişkenleri bu diziye aktarıp p değerleri ile SL'ı karşılaştırıp durum neticesinde değişkenleri çıkartıcam ya da bırakacağım.
X=np.append(arr = np.ones((22,1)).astype(int),values=veri,axis=1) #B0 değerlerini ekliyoruz.(1'lerden oluşan bir kolon)
X_l=veri.iloc[:,[0,1,2,3,4,5]].values

#Bağımlı değişken ile Bağımsız değişkenlerin bulunduğu veri kümesi arasında ki bağlantı sağlanıyor. 
#Yapılmak istenen Bu linear regression Modelindeki her kolonun boy'a olan etkisini ölçmektir.
r_ols = sm.OLS(endog=boy,exog=X_l).fit()
print(r_ols.summary()) #Sonuçta şimdilik dikkat edilmesi gereken konu. p-değerinin yazılı olduğu kolondur
'''
x1 0'a x2 1'e denk gelmektedir. Yapmamız gerken karşımıza çıkan p değeri ile PL'i karşılaştırmaktır.

Çıkan sonuçta x5'ait p-değeri 0.717 çıkmıştır(Bu sonuç değişiklik gösterebilir). Bu durumda Backward Elemination'a göre p-value 0.05'in üzerindeki
değişken eğitimden çıkartılır. O halde yapmamız gereken x5'i çıkartmak. Buda 4. Colondaki yaş alanına denk gelmektedir.
'''
#X_l=veri.iloc[:,[0,1,2,3,5]].values
#r_ols = sm.OLS(endog=boy,exog=X_l).fit()
#print(r_ols.summary())
'''
Çıkan sonuç kabul Backward Elemination'a göre kabul edilebilir bir durumdadır.
'''











































    
    

