# Görev 1:  Verilen değerlerin veri yapılarını inceleyiniz
import plistlib

x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1, 2, 3, 4]
d = {"Name": "Jake",
     "Age": 27,
     "Address": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science"}

print(type(x), type(y), type(z), type(a), type(b), type(c), type(l), type(d), type(t), type(s))


# Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
# kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into insight."

text = text.replace(",", " ").replace(".", " ").upper().split()
print(text)


# Görev 3:  Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
print(len(lst))
# Adım 2: Sıfırıncı ve onuncu indesteki elemanları çağırınız.
print(lst[0],lst[10])
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz
liste = lst[:4]
print(liste)
# Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)
print(lst)
# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("X")
print(lst)
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")
print(lst)


# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Adım1: Key değerlerine erişiniz.
print(dict.keys())
# Adım2: Value'lara erişiniz.
print(dict.values())
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13
print(dict)
# Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey", 24]
print(dict)
# Adım5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")
print(dict)


# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları
# ayrı listelere atayan ve bu listeleri return eden fonksiyonu yazınız.

l = [2, 13, 18, 93, 22]

def func(liste):
     even_list = []
     odd_list = []
     [even_list.append(i) if i % 2 == 0 else odd_list.append(i)  for i in liste]
     return even_list, odd_list

even_list, odd_list = func(l)
print(even_list, odd_list)


# Görev 6: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin
# isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz

import seaborn as sns
df = sns.load_dataset("car_crashes")
print(df.columns)

df.columns = ["NUM_"+i.upper() if df[i].dtype != "O" else i.upper() for i in df]
print(df.columns)


# Görev 7: List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no"
# BARINDIRMAYAN değişkenlerin sonuna "FLAG" yazınız.

import seaborn as sns
df = sns.load_dataset("car_crashes")
print(df.columns)

df.columns = [i.upper()+"_FLAG" if "no" not in i else i.upper() for i in df.columns]
print(df.columns)

# Görev 8: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden
# FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir data frame oluşturunuz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
print(df.columns)

og_list = ["abbrev", "no_previous"]

new_cols = [i for i in df.columns if i not in og_list]
new_df = df[new_cols]
print(new_df.head())
