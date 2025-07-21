import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💲", layout="centered")

st.title("Employee Salary Classification App💲")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", [
    "Male", "Female"
])

race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])

native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada",
    "India", "Puerto-Rico", "El-Salvador", "Cuba", "England",
    "Jamaica", "South", "China", "Italy", "Dominican-Republic",
    "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
    "Haiti", "Iran", "Taiwan", "Portugal", "Nicaragua",
    "Peru", "France", "Ecuador", "Ireland", "Cambodia",
    "Laos", "Thailand", "Scotland", "Yugoslavia", "Hungary",
    "Trinadad&Tobago", "Greece", "Hong", "Holand-Netherlands",
    "Others"
])

education_map = {
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 15
}

selected_education = st.sidebar.selectbox("Select your education level", list(education_map.keys()))
educational_num = education_map[selected_education]
# st.write(f"You selected: {selected_education} → Encoded as: {educational_num}")

marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])

relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Wife", "Not-in-family", "Own-child",
    "Unmarried", "Other-relative"
])

occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv"
])
workclass = st.sidebar.selectbox("WorkClass", [
    "Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov", "Others"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
#experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
capital_gain = st.sidebar.slider("Capital Gain", 0, 99999, 1000)
capital_loss = st.sidebar.slider("Capital Loss", 0, 4300, 500)

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender':[gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Encoding
from sklearn.preprocessing import LabelEncoder

# Recreate encoder and apply it (same categories as in training)
encoder = LabelEncoder()

input_df['workclass'] = encoder.fit_transform(input_df['workclass'])
input_df['marital-status'] = encoder.fit_transform(input_df['marital-status'])
input_df['occupation'] = encoder.fit_transform(input_df['occupation'])
input_df['relationship'] = encoder.fit_transform(input_df['relationship'])
input_df['race'] = encoder.fit_transform(input_df['race'])
input_df['gender'] = encoder.fit_transform(input_df['gender'])
input_df['native-country'] = encoder.fit_transform(input_df['native-country'])


# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    if prediction == 1:
        st.success("Predicted Income: >50K 🤑")
        st.balloons()
    else:
        st.warning("Predicted Income: <=50K 💼")

# # Batch prediction
# st.markdown("---")
# st.markdown("#### 📂 Batch Prediction")
# uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

# if uploaded_file is not None:
#     batch_data = pd.read_csv(uploaded_file)
#     st.write("Uploaded data preview:", batch_data.head())
#     batch_preds = model.predict(batch_data)
#     batch_data['PredictedClass'] = batch_preds
#     st.write("✅ Predictions:")
#     st.write(batch_data.head())
#     csv = batch_data.to_csv(index=False).encode('utf-8')
#     st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

