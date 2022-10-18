from contextlib import AsyncExitStack
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from project_Logistic import X_test

data = pd.read_csv("diabetes.csv")

data = data.drop({"Pregnancies","Outcome"},axis=1)

independant = data.drop(columns = "Glucose", axis = 1)
dependant = data["Glucose"]

X_train, X_test, y_train, y_test = train_test_split(independant,dependant,train_size=0.75)

regressor = LinearRegression()
regressor.fit(independant,dependant)

outcome = regressor.predict([[72,35,0,33.6,0.627,50]])

print("The glucose level is ",outcome)