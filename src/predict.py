import pickle
import pandas as pd

model = pickle.load(
    open("models/churn_model.pkl","rb")
)

sample = pd.DataFrame({
    'Age':[35],
    'MonthlyCharges':[65],
    'Tenure':[20],
    'SupportTickets':[4],
    'ContractType_One year':[0],
    'ContractType_Two year':[0],
    'InternetService_Fiber':[1]
})

prediction = model.predict(sample)

print(
    "Customer will churn"
    if prediction[0]==1
    else "Customer will stay"
)
