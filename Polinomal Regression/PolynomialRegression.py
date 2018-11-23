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

#linear regression(Doğrusal Model Oluşturma)
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(x,y)
#Görselleştirme
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()

#polinomial regression(Doğrusal Olmayan Model Oluşturma)
from sklearn.preprocessing import PolynomialFeatures #Herhangi bir sayıyı polinomal olarak ifade etmeye yarayan bir kütüphane
poly_reg= PolynomialFeatures(degree=2)#Oluşturduğum objenin 2.dereceden bir obje olduğunu belirttim.
#x değerimi polinomal bir değere dönüştürüyorum
x_poly=poly_reg.fit_transform(x)#(x'i) polinomal bir değere dönüştürüyorum.
lin_reg2=LinearRegression()  
lin_reg2.fit(x_poly,y)#polinominal değer ile fit ediyorum.Yine linear regresyon objesi içerisinde :) Burası önemli
#Görselleştirme
plt.scatter(x,y)
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)))#Aynı şekilde burada da görselleştirirken x'i polin
plt.show()

#polinomial regression
poly_reg3= PolynomialFeatures(degree=4)#Oluşturduğum objenin 4.dereceden bir obje olduğunu belirttim.
#x değerimi polinomal bir değere dönüştürüyorum
x_poly3=poly_reg3.fit_transform(x)#(x'i) polinomal bir değere dönüştürüyorum.
lin_reg3=LinearRegression()  
lin_reg3.fit(x_poly3,y)
#Görselleştirme
plt.scatter(x,y)
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(x)))#Aynı şekilde burada da görselleştirirken x'i polin
plt.show()
#tahminler

'''
Bir kişinin eğitim seviyelerine göre tahmini maaşları (Linear Regresyona göre)
'''
#tahmin et :)
print(lin_reg.predict(11)) #♠34716
print(lin_reg.predict(6.6)) #16923

'''Burada da polinomal regresyona göre bir tahmin yürütülüyor.'''
print(lin_reg2.predict(poly_reg.fit_transform(11))) #89000
print(lin_reg2.predict(poly_reg.fit_transform(6.6))) #8146