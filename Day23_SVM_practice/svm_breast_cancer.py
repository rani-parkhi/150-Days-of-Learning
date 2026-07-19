from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#Load dataset
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

df = pd.DataFrame(x , columns = cancer.feature_names)
df["Target"] = y

print(df.head())

#Explore dataset
print(df.shape)
print(df.info())
print(df.describe())
print(df["Target"].value_counts())
print(df.isnull().sum())

#Train-Test Split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

print(x_train.shape)
print(x_test.shape)

#Apply StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#Linear SVM
linear_svm = SVC(kernel="linear")
linear_svm.fit(x_train, y_train)
linear_pred = linear_svm.predict(x_test)

print("Linear SVM accuracy")
print(accuracy_score(y_test, linear_pred))
print(confusion_matrix(y_test, linear_pred))
print(classification_report(y_test, linear_pred))

#RBF SVM
rbf_svm = SVC(kernel="linear")
rbf_svm.fit(x_train, y_train)
rbf_pred = rbf_svm.predict(x_test)

print("RBF SVM accuracy")
print(accuracy_score(y_test, rbf_pred))
print(confusion_matrix(y_test, rbf_pred))
print(classification_report(y_test, rbf_pred))