import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

# Sample training data to simulate trained model
# (In real deployment, you can load the model from joblib/pickle)
data = pd.read_csv("student-mat.csv", sep=';')

# Add pass column
data['pass'] = data['G3'] >= 10
data['pass'] = data['pass'].map({True: 'yes', False: 'no'})
data['pass_binary'] = data['pass'].map({'yes': 1, 'no': 0})

# Features and targets
X = data[['G1', 'G2', 'failures', 'studytime']]
y_reg = data['G3']
y_cls = data['pass_binary']

# Train models
reg_model = LinearRegression().fit(X, y_reg)
cls_model = LogisticRegression().fit(X, y_cls)

# ----------------- Streamlit UI -----------------

st.title("🎓 Student Performance Predictor")
st.subheader("Predict Final Grade and Pass/Fail")

# Input values
g1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
g2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
studytime = st.selectbox("Study Time (1 = <2 hrs, 4 = >10 hrs)", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

# Predict when button clicked
if st.button("Predict"):
    user_data = np.array([[g1, g2, failures, studytime]])
    
    # Predict G3
    predicted_grade = reg_model.predict(user_data)[0]
    predicted_pass = cls_model.predict(user_data)[0]

    st.markdown("### 📊 Predicted Final Grade (G3): {:.2f}".format(predicted_grade))
    
    if predicted_pass == 1:
        st.success("✅ Status: Pass")
    else:
        st.error("❌ Status: Fail")
