import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide")
st.title("✈️ Airline Passenger Satisfaction ML Dashboard")

# FEATURES
features_main = [
    "Gender","Customer Type","Age","Type of Travel","Class",
    "Flight Distance","Inflight wifi service",
    "Departure/Arrival time convenient","Ease of Online booking",
    "Gate location","Food and drink","Online boarding",
    "Seat comfort","Inflight entertainment","On-board service",
    "Leg room service","Baggage handling","Checkin service",
    "Inflight service","Cleanliness","Departure Delay in Minutes",
    "Arrival Delay in Minutes"
]

features_genetic = [
    'Customer Type','Age','Type of Travel','Class',
    'Inflight wifi service','Departure/Arrival time convenient',
    'Ease of Online booking','Food and drink','Online boarding',
    'Seat comfort','On-board service','Leg room service',
    'Baggage handling','Checkin service','Inflight service',
    'Cleanliness','Arrival Delay in Minutes'
]

# DATA
train_data = pd.read_csv(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\data\train.csv")
test_data = pd.read_csv(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\data\test.csv")

all_data = pd.concat((train_data.loc[:, "Gender":"satisfaction"],
                      test_data.loc[:, "Gender":"satisfaction"]))

# PREPROCESS
all_data["Arrival Delay in Minutes"] = all_data["Arrival Delay in Minutes"].fillna(
    all_data["Arrival Delay in Minutes"].median()
)

le = LabelEncoder()
all_data["satisfaction"] = le.fit_transform(all_data["satisfaction"].str.strip())

for col in ["Gender", "Customer Type", "Type of Travel", "Class"]:
    all_data[col] = le.fit_transform(all_data[col].str.strip())

x_raw = all_data.drop("satisfaction", axis=1)
y = all_data["satisfaction"]

# SPLIT (UNSCALED)
x_Train, x_Test, y_Train, y_Test = train_test_split(
    x_raw, y, test_size=0.3, random_state=42
)

# MODELS
try:
    model_configs = {
        "Logistic Regression": {
            "model": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_lg.pkl", "rb")),
            "features": features_main
        },
        "Genetik": {
            "model": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_genetic.pkl", "rb")),
            "features": features_genetic
        },
        "Yapay Sinir Ağları": {
            "model": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_ysa.pkl", "rb")),
            "features": features_main
        },
        "Decision Tree": {
            "model": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_decision_trees_gini.pkl", "rb")),
            "features": features_main
        },
        "Gaussian NB": {
            "model": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_gnb.pkl", "rb")),
            "features": features_main
        },
        "Bernoulli NB": {
            "model": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_brn.pkl", "rb")),
            "features": features_main
        }
    }
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

# SIDEBAR
st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox("Select Model", list(model_configs.keys()))

selected_config = model_configs[selected_model_name]
model = selected_config["model"]
features = selected_config["features"]

# INPUT
st.subheader("🧾 Passenger Info")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 0, 100, 30)
    flight_distance = st.number_input("Flight Distance", value=1000)

with col2:
    delay = st.number_input("Delay (min)", value=0)
    wifi = st.slider("Wifi", 0, 5, 3)

with col3:
    food = st.slider("Food", 0, 5, 3)

input_dict = {
    "Gender": 0, "Customer Type": 0, "Age": age, "Type of Travel": 0, "Class": 0,
    "Flight Distance": flight_distance, "Inflight wifi service": wifi,
    "Departure/Arrival time convenient": 3, "Ease of Online booking": 3,
    "Gate location": 3, "Food and drink": food, "Online boarding": 3,
    "Seat comfort": 3, "Inflight entertainment": 3, "On-board service": 3,
    "Leg room service": 3, "Baggage handling": 3, "Checkin service": 3,
    "Inflight service": 3, "Cleanliness": 3,
    "Departure Delay in Minutes": delay,
    "Arrival Delay in Minutes": delay
}

# PREDICT
if st.button("🚀 Predict"):
    input_df = pd.DataFrame([input_dict])
    X_input = input_df[features]

    scaler_model = MinMaxScaler()
    scaler_model.fit(x_raw[features])
    X_scaled = scaler_model.transform(X_input)

    pred = model.predict(X_scaled)

    if len(pred.shape) > 1 or pred.dtype != int:
        pred = (pred > 0.5).astype(int).flatten()

    prediction = pred[0]

    st.subheader("🎯 Result")
    if prediction == 1:
        st.success("😊 SATISFIED")
    else:
        st.error("😞 NOT SATISFIED")

# MODEL COMPARISON
st.subheader("📊 Model Comparison")

model_names, accuracies, f1_scores = [], [], []

for name, config in model_configs.items():
    try:
        m = config["model"]
        feats = config["features"]

        X_test_model = x_Test[feats]

        scaler_model = MinMaxScaler()
        scaler_model.fit(x_raw[feats])
        X_test_scaled = scaler_model.transform(X_test_model)

        p = m.predict(X_test_scaled)

        if len(p.shape) > 1 or p.dtype != int:
            p = (p > 0.5).astype(int).flatten()

        acc = accuracy_score(y_Test, p)
        f1 = f1_score(y_Test, p)

        model_names.append(name)
        accuracies.append(acc)
        f1_scores.append(f1)

    except Exception as e:
        st.warning(f"{name} hata: {e}")

# GRAFİKLER
col1, col2 = st.columns(2)

with col1:
    if model_names:
        fig1, ax1 = plt.subplots(figsize=(10,5))
        sns.barplot(x=model_names, y=accuracies, ax=ax1)
        ax1.set_title("Accuracy")
        ax1.set_xticklabels(model_names,rotation=30,ha="right")
        st.pyplot(fig1)

with col2:
    if model_names:
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(model_names, f1_scores, marker="o")
        ax2.set_title("F1 Score")
        ax2.set_xticklabels(model_names,rotation=30,ha="right")
        st.pyplot(fig2)

# CONFUSION MATRIX
st.subheader("📉 Confusion Matrix")

try:
    X_test_model = x_Test[features]

    scaler_model = MinMaxScaler()
    scaler_model.fit(x_raw[features])
    X_test_scaled = scaler_model.transform(X_test_model)

    preds = model.predict(X_test_scaled)

    if len(preds.shape) > 1 or preds.dtype != int:
        preds = (preds > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_Test, preds)

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    st.pyplot(fig3)

except Exception as e:
    st.error(e)

# FEATURE IMPORTANCE
st.subheader("🌳 Feature Importance")

if selected_model_name == "Logistic Regression":
    importance = model.coef_[0]

elif hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

elif selected_model_name == "Yapay Sinir Ağları":
    importance=None
    st.info("ANN feature importance desteklenmez (SHAP gerekir)")

else:
    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(model, X_test_scaled, y_Test)
        importance = result.importances_mean
    except:
        st.warning("Bu model için importance hesaplanamadı")
        importance = None

fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_title("Feature Importance")

st.pyplot(fig)