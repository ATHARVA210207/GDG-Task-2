import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('StudentsPerformance.csv')
print("First 5 rows:")
print(data.head())
print("Shape:", data.shape)
data.info()

print("Missing values:")
print(data.isnull().sum())
data = data.fillna(data.mean(numeric_only=True))

X = data[['math score']]
y = data['reading score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("R2 Score:", r2)