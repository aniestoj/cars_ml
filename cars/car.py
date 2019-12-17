import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 1 - wczytanie danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
dataset = pd.read_csv(url, names=names)

# 1 - wypisanie pierwszych 5 rekordów
print(dataset.head(5))

# 1 - wywołać metodę describe
print(dataset.describe())

# 1 - wypisać liczbę atrybutów decyzyjnych
print(names)

# 4 - wykorzystanie LabelEncoder do zamiany na wartości liczbowe
encoder = LabelEncoder()
for i in dataset.columns:
    dataset[i] = encoder.fit_transform(dataset[i])

# 3 - wykresy, histogramy
sns.set_style("whitegrid")
sns.pairplot(dataset)
plt.show()

# 2 - podzielenie zbioru danych, na atrybuty decyzyjne i warunkowe
X = dataset.iloc[:, :6]  # atrybuty warunkowe
y = dataset["class"]  # atrybut decyzyjny

# 5 - podzielenie zbioru danych, na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 6 - algorytm losowy
random_class = np.random.choice([0, 1, 2, 3], len(y_test))
r_score = accuracy_score(y_test, random_class)

# 6 - Klasyfikator KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)

# 6 - Regresja logistyczna
lr = LogisticRegression(C=1)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)

# 6 - Drzewa decyzyjne
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)

# 7 - sprawdzić jakość klasyfikacji
print("Random: " + str(r_score))
print("KNN: " + str(knn_score))
print("LR: " + str(lr_score))
print("DT: " + str(dt_score))
print()

# 8 - Klasyfikator KNN
knn_results = []
for i in range(50):
    knn = KNeighborsClassifier(n_neighbors=i + 1)
    knn.fit(X_train, y_train)
    knn_results.append(knn.score(X_test, y_test))
data = pd.DataFrame({"y": knn_results, "x": range(1, 51)})
sns.lineplot(x='x', y='y', data=data).set_title('KNeighborsClassifier')
plt.show()

# 8 - Regresja logistyczna
lr_results = []
for i in range(10):
    lr = LogisticRegression(C=i + 1)
    lr.fit(X_train, y_train)
    lr_results.append(lr.score(X_test, y_test))
data = pd.DataFrame({"y": lr_results, "x": range(1, 11)})
sns.lineplot(x='x', y='y', data=data).set_title('Logistic Regression')
plt.show()

# 8 - Drzewa decyzyjne
dt_results = []
for i in range(10):
    dt = DecisionTreeClassifier(max_depth=i + 1)
    dt.fit(X_train, y_train)
    dt_results.append(dt.score(X_test, y_test))
data = pd.DataFrame({"y": dt_results, "x": range(1, 11)})
sns.lineplot(x='x', y='y', data=data).set_title('Decision Tree Classifier')
plt.show()

# 9 - Algorytm losowy
random_class = np.random.choice([0, 1, 2, 3], len(y_test))
r_score = accuracy_score(y_test, random_class)
print("Najlepsze parametry: ")
print("Random: " + str(r_score))

# 9 - Klasyfikator KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
print("KNN: " + str(knn_score))

# 9 - Regresja logistyczna
lr = LogisticRegression(C=1)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print("LR: " + str(lr_score))

# 9 - Drzewa decyzyjne
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)
print("DT: " + str(dt_score))
