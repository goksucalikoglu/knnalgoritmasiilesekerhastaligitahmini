from tkinter import Tk, messagebox, END, Label, Entry, Button

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("C:/Users/lenovo/PycharmProjects/knndiabet/diabetes.csv")
data.head()
print (data)
seker_hastalari = data[data.Outcome==1]
saglikli_insanlar= data[data.Outcome==0]

plt.scatter(saglikli_insanlar.Age, saglikli_insanlar.Glucose, color="green", label="sağlıklı", alpha = 0.2)
plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose, color="red", label="diabet hastası", alpha = 0.2)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

y= data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1)

x = (x_ham_veri -np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

print("Normalizasyon öncesi: \n")
print(x_ham_veri.head())

print("\n\n Normalizasyon sonrası")
print(x.head())

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=1)#yüzde 20 test
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
print("K=3  için başarı:",knn.score(x_test,y_test))

sayac=1
for k in range(1,11):
        knn_yeni=KNeighborsClassifier(n_neighbors = k)
        knn_yeni.fit(x_train,y_train)
        print(sayac," ","Doğruluk oranı:%",knn_yeni.score(x_test,y_test)*100)
        sayac +=1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)

ws= Tk()
ws.title("Şeker hastalığı olanlar")
ws.geometry('600x400')
ws['bg'] = '#ffbf00'


def printValue():
        Cocuk_sayisi = a1.get()
        Glikoz = a2.get()
        Kan_basinci = a3.get()
        Deri_kalinligi = a4.get()
        Insulin = a5.get()
        BMI = a6.get()
        Diabet_pedigree_fonksiyon = a7.get()
        Yas = a8.get()
        tahmin_listesi = []
        tahmin_listesi.append(Cocuk_sayisi)
        tahmin_listesi.append(Glikoz)
        tahmin_listesi.append(Kan_basinci)
        tahmin_listesi.append(Deri_kalinligi)
        tahmin_listesi.append(Insulin)
        tahmin_listesi.append(BMI)
        tahmin_listesi.append(Diabet_pedigree_fonksiyon)
        tahmin_listesi.append(Yas)
        Pred = []
        Pred.append(tahmin_listesi)
        print(tahmin_listesi)
        new_prediction = knn.predict(sc.transform(np.array(Pred)))
        if new_prediction == 0:
                new_prediction = 'Hasta Değil'
        else:
                new_prediction = 'Kişi Şeker Hastası'
        messagebox.showinfo("Sonuç", new_prediction)


def clear_text():
        a1.delete(0, END)
        a2.delete(0, END)
        a3.delete(0, END)
        a4.delete(0, END)
        a5.delete(0, END)
        a6.delete(0, END)
        a7.delete(0, END)
        a8.delete(0, END)


ws.wm_attributes('-transparentcolor', 'grey')

label = Label(ws, text="Çocuk Sayısı", bg='#ffbf00')
label.pack()
a1 = Entry(ws, width=7, font=('Arial 24'))
a1.pack(pady=15)
label = Label(ws, text="Glikoz", bg='#ffbf00')
label.pack()
a2 = Entry(ws, width=7, font=('Arial 24'))
a2.pack(pady=15)
label = Label(ws, text="Kan Basıncı", bg='#ffbf00')
label.pack()
a3 = Entry(ws, width=7, font=('Arial 24'))
a3.pack(pady=15)
label = Label(ws, text="Deri Kalınlığı", bg='#ffbf00')
label.pack()
a4 = Entry(ws, width=7, font=('Arial 24'))
a4.pack(pady=15)
label = Label(ws, text="İnsülin", bg='#ffbf00')
label.pack()
a5 = Entry(ws, width=7, font=('Arial 24'))
a5.pack(pady=15)
label = Label(ws, text="BMI", bg='#ffbf00')
label.pack()
a6 = Entry(ws, width=7, font=('Arial 24'))
a6.pack(pady=15)
label = Label(ws, text="Diabet Pedigree Değeri", bg='#ffbf00')
label.pack()
a7 = Entry(ws, width=7, font=('Arial 24'))
a7.pack(pady=15)
label = Label(ws, text="Yaş", bg='#ffbf00')
label.pack()
a8 = Entry(ws, width=7, font=('Arial 24'))
a8.pack(pady=15)

Button(
        ws,
        text="Hesapla",
        padx=10,
        pady=5,
        command=printValue
).pack()

Button(

        ws,
        text="Temizle",
        padx=10,
        pady=5,
        command=clear_text
).pack()

ws.mainloop()

















