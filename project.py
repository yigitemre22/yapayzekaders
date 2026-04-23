import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide")
st.title("✈️ Airline Passenger Satisfaction ML Dashboard")

# 1. MODELLERİ YÜKLE
# NOT: Eğer hata devam ederse bu .pkl dosyalarının içinde model olduğundan emin olun (sayı değil!)
try:
    models = {
        "Logistic Regression": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_lg.pkl", "rb")),
        "Yapay Sinir Ağları": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_ysa.pkl", "rb")),
        "Gaussian Naive Byes":pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_gnb.pkl","rb")),
        "Bernoulli Naive Byes":pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_brn.pkl","rb")),
        "Decision Trees Gini Index":pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_decision_trees_gini.pkl","rb"))
        #"Genetik": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_genetic.pkl", "rb"))
    }
except Exception as e:
    st.error(f"Model yükleme hatası: {e}. Lütfen .pkl dosyalarınızı kontrol edin.")

st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# 2. VERİ YÜKLEME VE SCALER HAZIRLIĞI (Tahmin öncesi gerekli)
train_data = pd.read_csv(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\data\train.csv")
test_data = pd.read_csv(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\data\test.csv")
all_data = pd.concat((train_data.loc[:, "Gender":"satisfaction"], test_data.loc[:, "Gender":"satisfaction"]))

# Eksik veriyi doldur ve encode et
all_data["Arrival Delay in Minutes"] = all_data["Arrival Delay in Minutes"].fillna(all_data["Arrival Delay in Minutes"].median())
le = LabelEncoder()
all_data["satisfaction"] = le.fit_transform(all_data["satisfaction"].str.strip())
for col in ["Gender", "Customer Type", "Type of Travel", "Class"]:
    all_data[col] = le.fit_transform(all_data[col].str.strip())

# Scaler'ı eğit
x_raw = all_data.drop("satisfaction", axis=1)
y = all_data["satisfaction"]
scaler = MinMaxScaler()
scaler.fit(x_raw) # Tüm veri üzerinden scaler'ı fit ediyoruz

# 3. KULLANICI GİRİŞLERİ
st.subheader("🧾 Passenger Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 0, 100, 30)
    flight_distance = st.number_input("Flight Distance", value=1000)
with col2:
    delay = st.number_input("Departure Delay (min)", value=0)
    wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
with col3:
    food = st.slider("Food Quality", 0, 5, 3)

# Tüm sütunları eğitim verisindeki sırayla oluşturun (Önemli!)
input_dict = {
    "Gender": 0, "Customer Type": 0, "Age": age, "Type of Travel": 0, "Class": 0,
    "Flight Distance": flight_distance, "Inflight wifi service": wifi,
    "Departure/Arrival time convenient": 3, "Ease of Online booking": 3,
    "Gate location": 3, "Food and drink": food, "Online boarding": 3,
    "Seat comfort": 3, "Inflight entertainment": 3, "On-board service": 3,
    "Leg room service": 3, "Baggage handling": 3, "Checkin service": 3,
    "Inflight service": 3, "Cleanliness": 3, "Departure Delay in Minutes": delay,
    "Arrival Delay in Minutes": delay
}

if st.button("🚀 Predict Satisfaction"):
    input_df = pd.DataFrame([input_dict])
    
    # GİRİŞİ ÖLÇEKLENDİR (Modelin anlayabilmesi için)
    input_scaled = scaler.transform(input_df)
    
    # TAHMİN
    raw_pred = model.predict(input_scaled)
    # YSA (Keras) olasılık döndürür, Sklearn direkt sınıf döndürebilir. İkisini de kapsayalım:
    prediction = (raw_pred > 0.5).astype(int).flatten()[0]

    st.subheader("🎯 Result")
    if prediction == 1:
        st.success("😊 Passenger is SATISFIED")
    else:
        st.error("😞 Passenger is NOT SATISFIED")

# 4. MODEL KARŞILAŞTIRMA (Test Verisi Üzerinden)
x_scaled = scaler.transform(x_raw)
from sklearn.model_selection import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

st.subheader("📊 Model Comparison")
model_names, accuracies, f1_scores = [], [], []

for name, m in models.items():
    try:
        p_proba = m.predict(x_Test)
        p = (p_proba > 0.5).astype(int).flatten()
        
        model_names.append(name)
        accuracies.append(accuracy_score(y_Test, p))
        f1_scores.append(f1_score(y_Test, p))
    except Exception as e:
        st.warning(f"{name} modeli çalıştırılamadı. Hata: {e}")

col1, col2 = st.columns(2)

with col1:
    if model_names:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        # Palette parametresi güncel sns sürümlerinde farklılık gösterebilir, 
        # eğer hata alırsanız palette="viridis" kısmını silebilirsiniz.
        sns.barplot(x=model_names, y=accuracies, ax=ax1) 
        ax1.set_title("Accuracy Comparison", fontsize=14)
        ax1.set_ylim(0, 1)
        
        # Değerleri barların üzerine yazdıran düzeltilmiş döngü
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
            
        st.pyplot(fig1)

with col2:
    if model_names:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(model_names, f1_scores, marker="o", linestyle="-", color="orange", markersize=8)
        ax2.set_title("F1 Score Comparison", fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

# 6. CONFUSION MATRIX (Seçili Model İçin)
st.subheader(f"📉 Confusion Matrix: {selected_model_name}")

try:
    # Seçili model ile test verisi üzerinde tahmin yap
    selected_pred_raw = model.predict(x_Test)
    selected_pred = (selected_pred_raw > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_Test, selected_pred)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Not Satisfied", "Satisfied"], 
                yticklabels=["Not Satisfied", "Satisfied"], ax=ax3)
    ax3.set_xlabel("Predicted Labels")
    ax3.set_ylabel("True Labels")
    st.pyplot(fig3)
except Exception as e:
    st.error(f"Matris çizilemedi: {e}")

# 7. FEATURE IMPORTANCE (Sadece Destekleyen Modeller İçin)
st.subheader("🌳 Feature Importance")

# Logistic Regression katsayılarını gösterebiliriz
if selected_model_name == "Logistic Regression" and hasattr(model, "coef_"):
    importance = model.coef_[0]
    feat_importances = pd.Series(importance, index=x_raw.columns)
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh', ax=ax4, color='teal')
    ax4.set_title("Top 10 Influential Features (Logistic Regression)")
    st.pyplot(fig4)
else:
    st.info("Bu model için özellik önemi gösterimi şu an desteklenmiyor (YSA için ekstra işlemler gerekir).")
# Grafikleri çiz (Aynı kod...)
# ... (plt.subplots kısımları buraya gelecek)