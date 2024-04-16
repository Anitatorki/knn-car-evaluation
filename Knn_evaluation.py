import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("car.data")


myPreprocessor = LabelEncoder()
buying = myPreprocessor.fit_transform(list(data["buying"]))
maint = myPreprocessor.fit_transform(list(data["maint"]))
door = myPreprocessor.fit_transform(list(data["door"]))
persons = myPreprocessor.fit_transform(list(data["persons"]))
lug_boot = myPreprocessor.fit_transform(list(data["lug_boot"]))
safety = myPreprocessor.fit_transform(list(data["safety"]))
clas = myPreprocessor.fit_transform(list(data["class"]))

predict = "class"

# data["door"].unique()

X = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(clas)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.1)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)


acc = model.score(x_test, y_test)
print(acc)