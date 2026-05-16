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

# ---------------- HANDLE MISSING VALUES ----------------

df = df.ffill()

# ---------------- DROP LOAN ID ----------------

if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# ---------------- SAVE ENCODERS ----------------

encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

# ---------------- FEATURES & TARGET ----------------

target_column = df.columns[-1]

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
    criterion="entropy",
    max_depth=8,
    min_samples_split=4,
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

applicant_income = st.text_input(
    "Applicant Income"
)

coapplicant_income = st.text_input(
    "Coapplicant Income"
)

# USER ENTERS FULL RUPEES

loan_amount_rupees = st.text_input(
    "Loan Amount (in Rupees)"
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

    if (
        applicant_income == "" or
        coapplicant_income == "" or
        loan_amount_rupees == ""
    ):

        st.error("Please enter all values")

    else:

        # ---------------- CONVERT VALUES ----------------

        applicant_income = int(applicant_income)

        coapplicant_income = int(coapplicant_income)

        # Convert rupees into dataset format

        loan_amount = int(loan_amount_rupees) / 1000

        # Convert years into months

        loan_amount_term = loan_amount_term * 12

        # ---------------- ENCODE VALUES ----------------

        gender = encoders["Gender"].transform([gender])[0]

        married = encoders["Married"].transform([married])[0]

        dependents = encoders["Dependents"].transform([dependents])[0]

        education = encoders["Education"].transform([education])[0]

        self_employed = encoders["Self_Employed"].transform([self_employed])[0]

        property_area = encoders["Property_Area"].transform([property_area])[0]

        # ---------------- CREATE INPUT DATA ----------------

        input_data = {}

        for col in X.columns:

            if col == "Gender":
                input_data[col] = gender

            elif col == "Married":
                input_data[col] = married

            elif col == "Dependents":
                input_data[col] = dependents

            elif col == "Education":
                input_data[col] = education

            elif col == "Self_Employed":
                input_data[col] = self_employed

            elif col == "ApplicantIncome":
                input_data[col] = applicant_income

            elif col == "CoapplicantIncome":
                input_data[col] = coapplicant_income

            elif col == "LoanAmount":
                input_data[col] = loan_amount

            elif col == "Loan_Amount_Term":
                input_data[col] = loan_amount_term

            elif col == "Credit_History":
                input_data[col] = credit_history

            elif col == "Property_Area":
                input_data[col] = property_area

        # ---------------- DATAFRAME ----------------

        input_df = pd.DataFrame([input_data])

        # ---------------- PREDICTION ----------------

        prediction = model.predict(input_df)

        probability = model.predict_proba(input_df)

        # Handle probability safely

        if probability.shape[1] == 2:

            rejected = probability[0][0] * 100

            approved = probability[0][1] * 100

        else:

            if prediction[0] == 1:
                approved = 100
                rejected = 0
            else:
                approved = 0
                rejected = 100

        # ---------------- OUTPUT ----------------

        if prediction[0] == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

        st.write("Approval Chance :", round(approved, 2), "%")

        st.write("Rejection Chance :", round(rejected, 2), "%")
