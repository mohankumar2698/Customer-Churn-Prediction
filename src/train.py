import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/customer_churn.csv")

df = pd.get_dummies(
    df,
    columns=["ContractType","InternetService"],
    drop_first=True
)

X = df.drop(["CustomerID","Churn"], axis=1)
y = df["Churn"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))

pickle.dump(
    model,
    open("models/churn_model.pkl","wb")
)

print("Model Saved")
