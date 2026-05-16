import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# TITLE
# -----------------------------
st.title("Predict Loan Approval")

st.write("Decision Tree Classifier Project")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("loan_data.csv")

# Fill missing values
df = df.ffill()

# -----------------------------
# ENCODE CATEGORICAL COLUMNS
# -----------------------------
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# FEATURES AND TARGET
# -----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# MODEL ACCURACY
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.success(f"Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# USER INPUTS
# -----------------------------
st.subheader("Enter Applicant Details")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    step=1000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    step=1000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    step=1000
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=1,
    step=1
)

credit_history = st.selectbox(
    "Credit History",
    [0, 1]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# -----------------------------
# MANUAL ENCODING
# -----------------------------
gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

if dependents == "0":
    dependents = 0
elif dependents == "1":
    dependents = 1
elif dependents == "2":
    dependents = 2
else:
    dependents = 3

education = 0 if education == "Graduate" else 1

self_employed = 1 if self_employed == "Yes" else 0

if property_area == "Urban":
    property_area = 2
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 0

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Loan Status"):

    input_data = [[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area
    ]]

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")