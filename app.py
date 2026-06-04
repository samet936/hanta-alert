import os
from datetime import datetime

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "veri.csv")
DATABASE_PATH = os.path.join(BASE_DIR, "hanta_alert.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = "hanta-alert-secret"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

ADMIN_USERNAME = "samet"
ADMIN_PASSWORD = "samet4040"

FEATURE_COLUMNS = [
    "fever", "cough", "headache", "muscle_pain", "shortness_breath",
    "fatigue", "nausea", "vomiting", "abdominal_pain", "diarrhea",
    "rodent_contact", "rodent_droppings", "storage_cleaning",
    "rural_visit", "camping_history", "age",
]

model = None
veri_sayisi = 0
MODEL_DOGRULUK = 0.0


class Analysis(db.Model):
    """Canlı hastalar için tasarlanmış, kimlik bilgilerini içeren analiz tablosu."""
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    surname = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(30), nullable=False)

    fever = db.Column(db.Integer, nullable=False)
    cough = db.Column(db.Integer, nullable=False)
    headache = db.Column(db.Integer, nullable=False)
    muscle_pain = db.Column(db.Integer, nullable=False)
    shortness_breath = db.Column(db.Integer, nullable=False)
    fatigue = db.Column(db.Integer, nullable=False)
    nausea = db.Column(db.Integer, nullable=False)
    vomiting = db.Column(db.Integer, nullable=False)
    abdominal_pain = db.Column(db.Integer, nullable=False)
    diarrhea = db.Column(db.Integer, nullable=False)

    rodent_contact = db.Column(db.Integer, nullable=False)
    rodent_droppings = db.Column(db.Integer, nullable=False)
    storage_cleaning = db.Column(db.Integer, nullable=False)
    rural_visit = db.Column(db.Integer, nullable=False)
    camping_history = db.Column(db.Integer, nullable=False)

    prediction = db.Column(db.String(30), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    analysis_date = db.Column(db.DateTime, default=datetime.now)


class TrainingData(db.Model):
    """Yapay zeka için ad, soyad ve cinsiyetten arındırılmış anonim laboratuvar tablosu."""
    __tablename__ = "training_data"

    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)

    fever = db.Column(db.Integer, nullable=False)
    cough = db.Column(db.Integer, nullable=False)
    headache = db.Column(db.Integer, nullable=False)
    muscle_pain = db.Column(db.Integer, nullable=False)
    shortness_breath = db.Column(db.Integer, nullable=False)
    fatigue = db.Column(db.Integer, nullable=False)
    nausea = db.Column(db.Integer, nullable=False)
    vomiting = db.Column(db.Integer, nullable=False)
    abdominal_pain = db.Column(db.Integer, nullable=False)
    diarrhea = db.Column(db.Integer, nullable=False)

    rodent_contact = db.Column(db.Integer, nullable=False)
    rodent_droppings = db.Column(db.Integer, nullable=False)
    storage_cleaning = db.Column(db.Integer, nullable=False)
    rural_visit = db.Column(db.Integer, nullable=False)
    camping_history = db.Column(db.Integer, nullable=False)

    actual_result = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)


def eski_csv_degerini_al(row, kolon, varsayilan=0):
    return int(row[kolon]) if kolon in row and pd.notna(row[kolon]) else varsayilan


def csvden_training_data_aktar():
    if TrainingData.query.count() > 0:
        return

    if not os.path.exists(DATASET_PATH):
        return

    df = pd.read_csv(DATASET_PATH, sep=";")

    for _, row in df.iterrows():
        db.session.add(TrainingData(
            age=eski_csv_degerini_al(row, "Yas", 35),
            fever=eski_csv_degerini_al(row, "Ates"),
            cough=eski_csv_degerini_al(row, "Oksuruk"),
            headache=eski_csv_degerini_al(row, "Bas_Agrisi"),
            muscle_pain=eski_csv_degerini_al(row, "Kas_Agrisi"),
            shortness_breath=eski_csv_degerini_al(row, "Nefes_Darligi"),
            fatigue=eski_csv_degerini_al(row, "Halsizlik"),
            nausea=eski_csv_degerini_al(row, "Bulanti"),
            vomiting=eski_csv_degerini_al(row, "Kusma"),
            abdominal_pain=eski_csv_degerini_al(row, "Karin_Agrisi"),
            diarrhea=eski_csv_degerini_al(row, "Ishal"),
            rodent_contact=eski_csv_degerini_al(row, "Kemirgen_Temasi"),
            rodent_droppings=eski_csv_degerini_al(row, "Kemirgen_Diskisi"),
            storage_cleaning=eski_csv_degerini_al(row, "Depo_Ahir_Temizligi"),
            rural_visit=eski_csv_degerini_al(row, "Kirsal_Bolge"),
            camping_history=eski_csv_degerini_al(row, "Kamp_Gecmisi"),
            actual_result=eski_csv_degerini_al(row, "Risk_Durumu"),
        ))

    db.session.commit()


def modeli_egit():
    global model, veri_sayisi, MODEL_DOGRULUK

    kayitlar = TrainingData.query.all()
    veri_sayisi = len(kayitlar)

    if veri_sayisi < 10:
        return

    X = pd.DataFrame([{c: getattr(k, c) for c in FEATURE_COLUMNS} for k in kayitlar])
    y = pd.Series([k.actual_result for k in kayitlar])

    df = X.copy()
    df["label"] = y

    majority = df[df["label"] == df["label"].mode()[0]]
    minority = df[df["label"] != df["label"].mode()[0]]

    if len(minority) > 0:
        minority_up = minority.sample(len(majority), replace=True, random_state=42)
        df = pd.concat([majority, minority_up])

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)
    MODEL_DOGRULUK = round(model.score(X_test, y_test) * 100, 2)


def evet_hayir_al(field):
    return int(request.form.get(field, 0))


def form_verilerini_al():
    age = int(request.form.get("age", 0))
    if age < 1 or age > 120:
        raise ValueError("Yaş 1-120 arası olmalı")

    return {
        "name": request.form.get("name", "").strip(),
        "surname": request.form.get("surname", "").strip(),
        "gender": request.form.get("gender", "").strip(),
        "age": age,
        "fever": evet_hayir_al("fever"),
        "cough": evet_hayir_al("cough"),
        "headache": evet_hayir_al("headache"),
        "muscle_pain": evet_hayir_al("muscle_pain"),
        "shortness_breath": evet_hayir_al("shortness_breath"),
        "fatigue": evet_hayir_al("fatigue"),
        "nausea": evet_hayir_al("nausea"),
        "vomiting": evet_hayir_al("vomiting"),
        "abdominal_pain": evet_hayir_al("abdominal_pain"),
        "diarrhea": evet_hayir_al("diarrhea"),
        "rodent_contact": evet_hayir_al("rodent_contact"),
        "rodent_droppings": evet_hayir_al("rodent_droppings"),
        "storage_cleaning": evet_hayir_al("storage_cleaning"),
        "rural_visit": evet_hayir_al("rural_visit"),
        "camping_history": evet_hayir_al("camping_history"),
    }


def ozellik_dataframe_olustur(veri):
    return pd.DataFrame([{c: veri[c] for c in FEATURE_COLUMNS}])


with app.app_context():
    db.create_all()
    csvden_training_data_aktar()
    modeli_egit()


@app.route("/")
def ana_sayfa():
    return render_template("index.html",
        veri_sayisi=veri_sayisi,
        model_dogruluk=MODEL_DOGRULUK,
        toplam_analiz=Analysis.query.count()
    )


@app.route("/risk-analizi", methods=["GET", "POST"])
def risk_analizi():
    sonuc = None

    if request.method == "POST":
        try:
            if model is None:
                raise ValueError("Model hazır değil")

            veri = form_verilerini_al()
            df = ozellik_dataframe_olustur(veri)

            proba = float(model.predict_proba(df)[0][1]) * 100

            # Klinik Skorlama Mekanizması (Güvenlik Önlemi)
            kritik_durum = (veri["shortness_breath"] == 1) or \
                           (veri["fever"] == 1 and (veri["rodent_contact"] == 1 or veri["rodent_droppings"] == 1))

            if proba >= 45 or kritik_durum:
                prediction = "Yüksek Risk"
                message = "Hantavirüs şüphesi/riski yüksek görünmektedir."
                recommendation = "ACİL: En yakın sağlık kuruluşuna başvurun ve kemirgen temasınızı bildirin!"
            else:
                prediction = "Düşük Risk"
                message = "Risk analizi tamamlandı. Aktif kritik semptomlar bulunamadı."
                recommendation = "Rutin takip: Belirtilerinizi izlemeye devam edin."

            sonuc = {
                "prediction": prediction,
                "probability": round(proba, 2),
                "message": message,
                "recommendation": recommendation,
                "veri_sayisi": veri_sayisi,
                "dogruluk": MODEL_DOGRULUK
            }

            db.session.add(Analysis(**veri, prediction=prediction, probability=proba))
            db.session.commit()

        except Exception as e:
            sonuc = {"hata": True, "mesaj": str(e)}

    return render_template("risk_analizi.html", sonuc=sonuc)


@app.route("/admin-giris", methods=["GET", "POST"])
def admin_giris():
    if request.method == "POST":
        if request.form["username"] == ADMIN_USERNAME and request.form["password"] == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("egitim_verisi_ekle_sayfasi"))
        flash("Hatalı giriş", "danger")

    return render_template("admin_giris.html")


@app.route("/admin-cikis")
def admin_cikis():
    session.clear()
    return redirect(url_for("ana_sayfa"))


@app.route("/egitim-verisi-ekle", methods=["GET", "POST"])
def egitim_verisi_ekle_sayfasi():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_giris"))

    if request.method == "POST":
        veri = form_verilerini_al()
        actual_result = int(request.form.get("actual_result"))

        # Eğitim verisi tablosuna kimlik bilgileri eklenmeden sadece gerekli parametreler kaydedilir
        db.session.add(TrainingData(
            age=veri["age"],
            fever=veri["fever"],
            cough=veri["cough"],
            headache=veri["headache"],
            muscle_pain=veri["muscle_pain"],
            shortness_breath=veri["shortness_breath"],
            fatigue=veri["fatigue"],
            nausea=veri["nausea"],
            vomiting=veri["vomiting"],
            abdominal_pain=veri["abdominal_pain"],
            diarrhea=veri["diarrhea"],
            rodent_contact=veri["rodent_contact"],
            rodent_droppings=veri["rodent_droppings"],
            storage_cleaning=veri["storage_cleaning"],
            rural_visit=veri["rural_visit"],
            camping_history=veri["camping_history"],
            actual_result=actual_result
        ))
        db.session.commit()

        modeli_egit()

        flash("Eklendi ve model eğitildi", "success")
        return redirect(url_for("egitim_verisi_ekle_sayfasi"))

    return render_template("egitim_verisi_ekle.html",
        veri_sayisi=veri_sayisi,
        model_dogruluk=MODEL_DOGRULUK
    )


@app.route("/kayitlar")
def kayitlar():
    return render_template("kayitlar.html",
        kayitlar=Analysis.query.order_by(Analysis.analysis_date.desc()).all()
    )


@app.route("/kayit-sil/<int:kayit_id>", methods=["POST"])
def kayit_sil(kayit_id):
    db.session.delete(Analysis.query.get_or_404(kayit_id))
    db.session.commit()
    return redirect(url_for("kayitlar"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)