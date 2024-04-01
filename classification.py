import pandas as pd
import sklearn 
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, model_selection, preprocessing
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("loan_data_1.csv")
df = pd.DataFrame(df.dropna())
df.drop(columns=["Unnamed: 0", "Loan_ID"], inplace=True)

df.replace({"Male": 0, "Female": 1}, inplace=True)

df.replace({"No": 0, "Yes": 1}, inplace=True)

filt = df["Dependents"] != "3+"
df = df[filt]
df["Dependents"] = df["Dependents"].astype(int)

df.replace({"Graduate": 1, "Not Graduate": 0}, inplace=True)

df.replace({"Rural": 0, "Urban": 1, "Semiurban": 2}, inplace=True)

df.replace({"Y": 1, "N": 0}, inplace=True)

x = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]



x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x, y)
acc = model.score(x_test, y_test)
print(acc)