from copy import deepcopy
import imp
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_csv("diabetes.csv")
data = data.drop(columns="Pregnancies", axis=1)
# print(data.head())

independant = data.drop(columns="Outcome", axis=1)
dependant = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(independant,dependant,train_size=0.75)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train,y_train)


# print(data.isnull().value_counts())
outcome_logistic = classifier.predict([[148,72,35,0,33.6,0.627,50]])

if outcome_logistic>0:
    print("The patient has diabetes")
else:
    print("The patient does not have diabetes")



