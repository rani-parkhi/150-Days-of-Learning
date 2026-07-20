from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Load dataset
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target
df = pd.DataFrame(x , columns=cancer.feature_names)
df["Target"] = y

#Train-Test split
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Apply StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Train Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_pred = nb.predict(x_test)

#Compare models

#Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

#SVM
svm = SVC(kernel="rbf")
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)

#Comparison
Comparison = pd.DataFrame({
    "Model": [
        "Logistic regression",
        "KNN",
        "SVM",
        "Gaussian Naive Bayes"
    ],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, nb_pred)
    ]
})

print(Comparison)