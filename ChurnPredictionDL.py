import numpy as np
import pandas as pd

df = pd.read_csv("churnPredictionDATASET\\Churn_Modelling.csv")
print("Dataset loaded successfully!")
print("First 5 rows of the dataset:")
print(df.head())
print("Dataset information:")
print(df.info())
print("Dataset description:")
print(df.describe())
print("Dataset shape:")
print(df.shape)
print("Duplicated rows:")
print(df.duplicated().sum())
print("exited employess:")
print(df["Exited"].value_counts())
print("geographical distribution of customers:")
print(df["Geography"].value_counts())
print("gender distribution of customers:")
print(df["Gender"].value_counts())

df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

print("First 5 rows of the cleaned dataset:")
print(df.head())

print("encoding categorical variables...")
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

print(df.head())


# splitting the dataset into features and target variable
X = df.drop(["Exited"], axis=1)
y = df["Exited"]



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)


print("Scaling the features...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Features scaled successfully!")
print(x_train_scaled)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(11, activation="relu", input_dim=11))
model.add(Dense(11, activation="relu"))

model.add(Dense(1, activation="sigmoid"))
print("Model summary:")
print(model.summary())



model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train_scaled, y_train, epochs=100,validation_split=0.2)

print("First layer weights:")
print(model.layers[0].get_weights())


model.predict(x_test_scaled)

y_log=model.predict(x_test_scaled)
y_pred = (y_log > 0.5).astype(int)

print("Predicted values:")
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)


history_df = history.history
print("Training history:")
print(history_df)

import matplotlib.pyplot as plt
plt.plot(history.history["loss"],color="blue", label="Training Loss")
plt.plot(history.history["val_loss"],color="orange", label="Validation Loss")
plt.plot(history.history["accuracy"],color="green", label="Training Accuracy")
plt.plot(history.history["val_accuracy"],color="red", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.title("Model Training History")
plt.legend()
plt.show()