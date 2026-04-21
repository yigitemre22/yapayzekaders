import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide")

st.title("✈️ Airline Passenger Satisfaction ML Dashboard")

# =========================
# MODELS LOAD
# =========================
models = {
    "Logistic Regression": pickle.load(open(r"C:\Users\yigit\OneDrive\Masaüstü\yapayzeka\model_lg.pkl", "rb")),
}

# ⚠️ Eğer training sırasında get_dummies yaptıysan bu şart:
# trained_columns = pickle.load(open("columns.pkl", "rb"))
# (şimdilik placeholder)
trained_columns = None

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[selected_model_name]

# =========================
# INPUTS (UI)
# =========================
st.subheader("🧾 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    flight_distance = st.number_input("Flight Distance", value=1000)

with col2:
    delay = st.number_input("Departure Delay (min)", value=0)
    wifi = st.slider("Inflight Wifi Service", 0, 5, 3)

with col3:
    food = st.slider("Food Quality", 0, 5, 3)

# =========================
# FIXED INPUT (KRİTİK KISIM)
# =========================
input_dict = {
    "Gender": 0,
    "Customer Type": 2,
    "Age": age,
    "Type of Travel": 1,
    "Class": 1,
    "Flight Distance": flight_distance,
    "Inflight wifi service": wifi,
    "Departure/Arrival time convenient": 3,
    "Ease of Online booking": 4,
    "Gate location": 3,
    "Food and drink": food,
    "Online boarding": 5,
    "Seat comfort": 4,
    "Inflight entertainment": 4,
    "On-board service": 4,
    "Leg room service": 4,
    "Baggage handling": 4,
    "Checkin service": 4,
    "Inflight service": 4,
    "Cleanliness": 4,
    "Departure Delay in Minutes": delay,
    "Arrival Delay in Minutes": delay
}

# =========================
# PREDICTION FIX
# =========================
if st.button("🚀 Predict Satisfaction"):

    # 1. dict → dataframe
    input_df = pd.DataFrame([input_dict])

    # 2. Eğer training one-hot yaptıysan aç:
    # input_df = pd.get_dummies(input_df)
    # input_df = input_df.reindex(columns=trained_columns, fill_value=0)

    # 3. PREDICT (ARTIK HATA YOK)
    prediction = model.predict(input_df)

    st.subheader("🎯 Result")

    if prediction[0] == 1:
        st.success("😊 Passenger is SATISFIED")
    else:
        st.error("😞 Passenger is NOT SATISFIED")

# =========================
# MOCK DATA (SADECE DEMO)
# =========================
X_test = np.random.rand(100, 22)
y_test = np.random.randint(0, 2, 100)

# =========================
# MODEL COMPARISON
# =========================
st.subheader("📊 Model Comparison")

model_names = []
accuracies = []
f1_scores = []

for name, m in models.items():
    pred = m.predict(X_test)

    model_names.append(name)
    accuracies.append(accuracy_score(y_test, pred))
    f1_scores.append(f1_score(y_test, pred))

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.bar(model_names, accuracies)
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(model_names, f1_scores, marker="o")
    ax.set_title("F1 Score Comparison")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📉 Confusion Matrix")

selected_pred = model.predict(X_test)
cm = confusion_matrix(y_test, selected_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("🌳 Feature Importance")

if selected_model_name == "Random Forest" and hasattr(model, "feature_importances_"):
    features = ["Age", "Distance", "Delay", "Wifi", "Food"]

    fig, ax = plt.subplots()
    ax.barh(features, model.feature_importances_)
    ax.set_title("Feature Importance")

    st.pyplot(fig)
else:
    st.info("Feature importance only available for tree-based models like Random Forest.")