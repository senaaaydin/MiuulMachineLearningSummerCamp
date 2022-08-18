# Görev 1: Aşağıdaki Soruları Yanıtlayınız

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("Miuul/datasets/persona.csv")
df.head()
df.describe().T

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].unique()

# Soru 3: Kaç unique PRICE vardır?

df["PRICE"].unique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?


df.groupby("COUNTRY")["PRICE"].sum()

# Soru 7: SOURCE türlerine göre satış sayıları nedir?

df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].mean()

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE")["PRICE"].mean()

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.pivot_table("PRICE", "COUNTRY", "SOURCE")


# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()

# Görev 3: Çıktıyı PRICE’a göre sıralayınız.

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.

df.index
agg_df = agg_df.reset_index()

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=["0_18", "19_23", "24_30", "31_40", "41_70"])

# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

agg_df["customers_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].apply("_".join, axis=1)
persona = agg_df[["customers_level_based", "PRICE"]]
persona = pd.DataFrame(persona.groupby("customers_level_based")["PRICE"].mean())


# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.

agg_df["SEGMENT"] = pd.cut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"PRICE":["mean", "max", "sum"]})

persona["SEGMENT"] = pd.cut(persona["PRICE"], 4, labels=["D", "C", "B", "A"])
persona.groupby("SEGMENT").agg({"PRICE":["mean", "max", "sum"]})

# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_costumer1 = "tur_android_female_31_40"
agg_df[agg_df["customers_level_based"] == new_costumer1]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_costumer2 = "fra_ios_female_31_40"
agg_df[agg_df["customers_level_based"] == new_costumer2]


