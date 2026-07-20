from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Loading dataset
wine = load_wine()

x = wine.data
y = wine.target

df = pd.DataFrame(x , columns=wine.feature_names)
df["Target"] = y

print(df.head())

#Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

print(x_train.shape)
print(x_test.shape)

#Apply StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

k_values = [1,3,5,7,9]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train , y_train)

    pred = model.predict(x_test)

    acc = accuracy_score(y_test , pred)

    print(f"k={k} Accuracy={acc:.4f}")