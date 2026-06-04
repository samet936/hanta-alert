import os
import random

import pandas as pd


random.seed(42)

DATASET_DIR = "datasets"
DATASET_PATH = os.path.join(DATASET_DIR, "veri.csv")

os.makedirs(DATASET_DIR, exist_ok=True)

kayitlar = []

for _ in range(500):
    yas = random.randint(5, 88)

    kemirgen_temasi = random.choices([0, 1], weights=[45, 55])[0]
    ates = random.choices([0, 1], weights=[30, 70])[0]
    bas_agrisi = random.choices([0, 1], weights=[35, 65])[0]
    kas_agrisi = random.choices([0, 1], weights=[25, 75])[0]
    oksuruk = random.choices([0, 1], weights=[55, 45])[0]
    nefes_darligi = random.choices([0, 1], weights=[65, 35])[0]
    halsizlik = random.choices([0, 1], weights=[30, 70])[0]

    puan = 0
    puan += 25 if kemirgen_temasi else 0
    puan += 15 if ates else 0
    puan += 10 if bas_agrisi else 0
    puan += 10 if kas_agrisi else 0
    puan += 10 if oksuruk else 0
    puan += 25 if nefes_darligi else 0
    puan += 10 if halsizlik else 0

    if yas >= 60:
        puan += 10

    risk_durumu = 1 if puan >= 60 else 0

    kayitlar.append({
        "Ates": ates,
        "Oksuruk": oksuruk,
        "Bas_Agrisi": bas_agrisi,
        "Kas_Agrisi": kas_agrisi,
        "Nefes_Darligi": nefes_darligi,
        "Halsizlik": halsizlik,
        "Kemirgen_Temasi": kemirgen_temasi,
        "Yas": yas,
        "Risk_Durumu": risk_durumu
    })

df = pd.DataFrame(kayitlar)
df.to_csv(DATASET_PATH, index=False, sep=";")

print("Veri seti başarıyla oluşturuldu.")
print(f"Dosya yolu: {DATASET_PATH}")
print(f"Kayıt sayısı: {len(df)}")