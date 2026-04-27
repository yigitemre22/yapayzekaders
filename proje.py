import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide")
st.title("✈️ Airline Passenger Satisfaction ML Dashboard")

# DATA
train_data = pd.read_csv(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\data\train.csv")
test_data  = pd.read_csv(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\data\test.csv")

all_data = pd.concat((
    train_data.loc[:, "Gender":"satisfaction"],
    test_data.loc[:,  "Gender":"satisfaction"]
)).reset_index(drop=True)

# PREPROCESS
all_data["Arrival Delay in Minutes"] = all_data["Arrival Delay in Minutes"].fillna(
    all_data["Arrival Delay in Minutes"].median()
)

le = LabelEncoder()
all_data["satisfaction"] = le.fit_transform(all_data["satisfaction"].str.strip())
for col in ["Gender", "Customer Type", "Type of Travel", "Class"]:
    all_data[col] = le.fit_transform(all_data[col].str.strip())

x_raw = all_data.drop("satisfaction", axis=1)
y     = all_data["satisfaction"]

x_Train, x_Test, y_Train, y_Test = train_test_split(
    x_raw, y, test_size=0.3, random_state=42
)

# MODELS — her pkl artık {"model", "scaler", "features"} içeriyor
try:
    model_configs = {}
    model_files = {
        "Logistic Regression":  r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_lg.pkl",
        "Genetik":              r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_genetic.pkl",
        "Yapay Sinir Ağları":   r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_ysa.pkl",
        "Decision Tree":        r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_decision_trees_gini.pkl",
        "Gaussian NB":          r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_gnb.pkl",
        "Bernoulli NB":         r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_brn.pkl",
    }

    for name, path in model_files.items():
        loaded = pickle.load(open(path, "rb"))
        model_configs[name] = {
            "model":    loaded["model"],
            "scaler":   loaded["scaler"],
            "features": loaded["features"],
        }

except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

# SIDEBAR
st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Select Model", list(model_configs.keys()))

selected_config = model_configs[selected_model_name]
model    = selected_config["model"]
scaler   = selected_config["scaler"]
features = selected_config["features"]

# INPUT
st.subheader("🧾 Passenger Info")

col1, col2, col3 = st.columns(3)

with col1:
    gender          = st.selectbox("Gender", ["Female", "Male"])
    customer_type   = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    age             = st.number_input("Age", 0, 100, 30)
    flight_distance = st.number_input("Flight Distance", value=1000)
    dep_delay       = st.number_input("Departure Delay (min)", value=0)
    arr_delay       = st.number_input("Arrival Delay (min)", value=0)

with col2:
    travel_type     = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class    = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    wifi            = st.slider("Inflight Wifi Service", 0, 5, 3)
    food            = st.slider("Food and Drink", 0, 5, 3)
    seat            = st.slider("Seat Comfort", 0, 5, 3)
    entertainment   = st.slider("Inflight Entertainment", 0, 5, 3)
    cleanliness     = st.slider("Cleanliness", 0, 5, 3)

with col3:
    dep_arr_time    = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)
    online_booking  = st.slider("Ease of Online Booking", 0, 5, 3)
    gate_location   = st.slider("Gate Location", 0, 5, 3)
    online_boarding = st.slider("Online Boarding", 0, 5, 3)
    onboard_service = st.slider("On-board Service", 0, 5, 3)
    leg_room        = st.slider("Leg Room Service", 0, 5, 3)
    baggage         = st.slider("Baggage Handling", 0, 5, 3)
    checkin         = st.slider("Checkin Service", 0, 5, 3)
    inflight_svc    = st.slider("Inflight Service", 0, 5, 3)

# Encode kategorik değerler (eğitimde kullanılan sırayla aynı)
gender_enc        = 1 if gender == "Male" else 0
customer_type_enc = 0 if customer_type == "Loyal Customer" else 1
travel_type_enc   = 1 if travel_type == "Personal Travel" else 0
class_enc         = {"Business": 0, "Eco": 1, "Eco Plus": 2}[travel_class]

input_dict = {
    "Gender":                           gender_enc,
    "Customer Type":                    customer_type_enc,
    "Age":                              age,
    "Type of Travel":                   travel_type_enc,
    "Class":                            class_enc,
    "Flight Distance":                  flight_distance,
    "Inflight wifi service":            wifi,
    "Departure/Arrival time convenient":dep_arr_time,
    "Ease of Online booking":           online_booking,
    "Gate location":                    gate_location,
    "Food and drink":                   food,
    "Online boarding":                  online_boarding,
    "Seat comfort":                     seat,
    "Inflight entertainment":           entertainment,
    "On-board service":                 onboard_service,
    "Leg room service":                 leg_room,
    "Baggage handling":                 baggage,
    "Checkin service":                  checkin,
    "Inflight service":                 inflight_svc,
    "Cleanliness":                      cleanliness,
    "Departure Delay in Minutes":       dep_delay,
    "Arrival Delay in Minutes":         arr_delay,
}

# PREDICT
if st.button("🚀 Predict"):
    input_df = pd.DataFrame([input_dict])
    X_input  = input_df[features]           # modelin kullandığı feature'lar
    X_scaled = scaler.transform(X_input)    # modelin kendi scaler'ı — fit() YOK

    pred = model.predict(X_scaled)

    if pred.ndim > 1 or pred.dtype != int:
        pred = (pred > 0.5).astype(int).flatten()

    st.subheader("🎯 Result")
    if pred[0] == 1:
        st.success("😊 SATISFIED")
    else:
        st.error("😞 NOT SATISFIED")

# MODEL COMPARISON
st.subheader("📊 Model Comparison")

model_names, accuracies, f1_scores = [], [], []

for name, config in model_configs.items():
    try:
        m        = config["model"]
        feats    = config["features"]
        sc       = config["scaler"]

        X_test_scaled = sc.transform(x_Test[feats])

        p = m.predict(X_test_scaled)
        if p.ndim > 1 or p.dtype != int:
            p = (p > 0.5).astype(int).flatten()

        model_names.append(name)
        accuracies.append(accuracy_score(y_Test, p))
        f1_scores.append(f1_score(y_Test, p))

    except Exception as e:
        st.warning(f"{name} hata: {e}")

col1, col2 = st.columns(2)

with col1:
    if model_names:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=model_names, y=accuracies, ax=ax1)
        ax1.set_title("Accuracy")
        ax1.set_xticklabels(model_names, rotation=30, ha="right")
        st.pyplot(fig1)

with col2:
    if model_names:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(model_names, f1_scores, marker="o")
        ax2.set_title("F1 Score")
        ax2.set_xticklabels(model_names, rotation=30, ha="right")
        st.pyplot(fig2)

# CONFUSION MATRIX
st.subheader("📉 Confusion Matrix")

try:
    X_test_scaled = scaler.transform(x_Test[features])
    preds = model.predict(X_test_scaled)

    if preds.ndim > 1 or preds.dtype != int:
        preds = (preds > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_Test, preds)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

except Exception as e:
    st.error(e)

# FEATURE IMPORTANCE
st.subheader("🌳 Feature Importance")

importance = None

if selected_model_name == "Logistic Regression":
    importance = model.coef_[0]

elif hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

elif selected_model_name == "Yapay Sinir Ağları":
    st.info("ANN feature importance desteklenmez (SHAP gerekir)")

else:
    try:
        X_test_scaled = scaler.transform(x_Test[features])
        result = permutation_importance(model, X_test_scaled, y_Test, n_repeats=5, random_state=42)
        importance = result.importances_mean
    except Exception as e:
        st.warning(f"Importance hesaplanamadı: {e}")

if importance is not None:
    fig4, ax4 = plt.subplots(figsize=(8, len(features) * 0.35 + 1))
    ax4.barh(features, importance)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)


