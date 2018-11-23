#Support Vector Regression 
'''
Margin değerlerini tanımlıyor ve bu margin değerlerine giren max noktayı 
elde edebileceği min margin değerine sahip margini içeren fonksiyonu almayı amaçlıyor 
şayet birden falza doğru çizilebliyorsa bu doğrulardan min margin değerine sahip aynı noktaları içine 
alabilecek değeri elde etmeye çalışıyor.
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

#verilerin olceklenmesi 
from sklearn.preprocessing import StandardScaler
###SVR'Da ölçeklendirme çok önemlidir.
sc1 = StandardScaler()#orta değeri 0 olarak kabul eder
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

#SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf') #Radyal Bases func.  polinomial-radial ile değişebilir.Burada kernel'in doğru seçilmesi çok önemlidir!!!!
svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli,color='red')#data point
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')#Her bir data point için 

print(svr_reg.predict(11)) #89000
print(svr_reg.predict(6.6)) #8146































