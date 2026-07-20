from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Load dataset
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target
df = pd.DataFrame(x , columns=cancer.feature_names)
df["Target"] = y
print(df.head)

#Explore dataset
#Display number of rows and columns
print(df.shape)

#Show dataset information
print(df.info())

#Display statistical summary
print(df.describe())

#Count samples in each class
print(df["Target"].value_counts())

#Check missing values
print(df.isnull().sum())

#Train-Test split
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

#Train Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_pred = nb.predict(x_test)

#Evaluate
print("Accuracy")
print(accuracy_score(y_test, nb_pred))

print()

print("Confusion Matrix")
print(confusion_matrix(y_test, nb_pred))

print()

print(classification_report(y_test, nb_pred))

