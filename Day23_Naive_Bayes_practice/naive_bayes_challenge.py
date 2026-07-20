from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Load dataset
wine = load_wine()

x = wine.data
y = wine.target
df = pd.DataFrame(x , columns=wine.feature_names)
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

#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

Comparison = pd.DataFrame({
    "Model": [
        "Gaussian Naive Bayes",
        "KNN"
    ],
    "Accuracy":[
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, knn_pred)
    ]
})

print(Comparison)