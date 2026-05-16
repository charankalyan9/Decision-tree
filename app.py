import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------------- TITLE ----------------

st.title("Loan Prediction using Decision Tree")

st.write("Predict Loan Approval")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("loan_prediction.csv")

# ---------------- CLEAN COLUMN NAMES ----------------

df.columns = df.columns.str.strip()

# ---------------- HANDLE MISSING VALUES ----------------

df = df.ffill()

# ---------------- DROP UNNECESSARY COLUMNS ----------------

for col in df.columns:

    if "loan_id" in col.lower() or "unnamed" in col.lower():

        df.drop(col, axis=1, inplace=True)

# ---------------- FIND TARGET COLUMN ----------------

target_column = None

for col in df.columns:

    if "status" in col.lower():

        target_column = col
        break

# ---------------- LABEL ENCODING ----------------

encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

# ---------------- FEATURES & TARGET ----------------

X = df.drop(target_column, axis=1)

y = df[target_column]

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ----------------

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- USER INPUTS ----------------

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
    step=1
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    step=1
)

loan_amount = st.number_input(
    "Loan Amount (in Rupees)",
    min_value=0,
    step=1
)

loan_amount_term = st.number_input(
    "Loan Amount Term (Years)",
    min_value=1,
    step=1
)

credit_history = st.selectbox(
    "Credit History",
    [0, 1]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

# ---------------- PREDICT ----------------

if st.button("Predict"):

    # Convert loan amount to dataset scale

    loan_amount = loan_amount / 1000

    # Convert years to months

    loan_amount_term = loan_amount_term * 12

    # Encode values

    gender = encoders["Gender"].transform([gender])[0]

    married = encoders["Married"].transform([married])[0]

    dependents = encoders["Dependents"].transform([dependents])[0]

    education = encoders["Education"].transform([education])[0]

    self_employed = encoders["Self_Employed"].transform([self_employed])[0]

    property_area = encoders["Property_Area"].transform([property_area])[0]

    # ---------------- CREATE INPUT ----------------

    input_df = pd.DataFrame([[
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
    ]], columns=X.columns)

    # ---------------- PREDICT ----------------

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    approved = probability[0][1] * 100

    rejected = probability[0][0] * 100

    # ---------------- OUTPUT ----------------

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.write("Approval Chance :", round(approved, 2), "%")

    st.write("Rejection Chance :", round(rejected, 2), "%")