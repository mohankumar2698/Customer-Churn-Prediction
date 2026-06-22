import streamlit as st
import pickle
import pandas as pd

model = pickle.load(
    open("../models/churn_model.pkl","rb")
)

st.title("Customer Churn Prediction")

age = st.slider("Age",18,70,30)
charges = st.slider("Monthly Charges",20,150,60)
tenure = st.slider("Tenure",1,100,20)
tickets = st.slider("Support Tickets",0,10,2)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month","One year","Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL","Fiber"]
)

one_year = 1 if contract=="One year" else 0
two_year = 1 if contract=="Two year" else 0
fiber = 1 if internet=="Fiber" else 0

data = pd.DataFrame({
    'Age':[age],
    'MonthlyCharges':[charges],
    'Tenure':[tenure],
    'SupportTickets':[tickets],
    'ContractType_One year':[one_year],
    'ContractType_Two year':[two_year],
    'InternetService_Fiber':[fiber]
})

if st.button("Predict"):
    result = model.predict(data)

    if result[0]==1:
        st.error("Customer likely to Churn")
    else:
        st.success("Customer likely to Stay")
