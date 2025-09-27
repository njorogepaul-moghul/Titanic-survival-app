import streamlit as st
import pickle
import numpy as np

st.title("🚢 Titanic Survival Prediction")
st.write("Choose a model and enter passenger details below:")

# Model selection
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model_file = "titanic_model.pkl"
else:
    model_file = "titanic_rf_model.pkl"

# Load the chosen model
with open(model_file, "rb") as f:
    model = pickle.load(f)


# User inputs

sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=30.0)

who = st.selectbox("Category", ["man", "woman", "child"])
alone = st.selectbox("Traveling Alone?", ["yes", "no"])

embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])


# Feature engineering
family_size = sibsp + parch + 1
is_child = 1 if age < 18 else 0
fare_per_person = fare / family_size if family_size > 0 else fare

# Label encoding (same as training)
sex = 1 if sex == "female" else 0   # male=0, female=1
alone = 1 if alone == "yes" else 0

who_map = {"man": 0, "woman": 1, "child": 2}
who = who_map[who]

# Embarked dummy
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Class dummy
class_Second = 1 if pclass == "Second" else 0
class_Third = 1 if pclass == "Third" else 0

# Arrange features in correct order
features = np.array([[
    sex, age, sibsp, parch, fare,
    who, alone,
    embarked_Q, embarked_S,
    class_Second, class_Third,
    family_size, is_child, fare_per_person
]])


# Prediction

if st.button("Predict Survival"):
    prediction = model.predict(features)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    st.subheader(f"{model_choice} Prediction: {result}")
