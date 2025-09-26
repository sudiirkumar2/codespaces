import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('./ML_lab/weather.csv')
df = df.dropna()

y = df['Weather']
X = df.drop(['Date/Time', 'Weather'], axis=1)

y_enc = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)

# KNN classifier (k=5 by default)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Test set predictions:', y_pred)
print('Test set accuracy:', accuracy_score(y_test, y_pred))