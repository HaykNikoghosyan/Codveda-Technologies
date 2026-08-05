
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('4) house Prediction Data Set.csv', sep=r'\s+', header=None)

X = df.iloc[:, [5]]
y =df.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(f"R-squared: (R2 Score): {r2_score(y_test,y_pred):.4f}")
print(f"Mean Squared Error(MSE): {mean_squared_error(y_test,y_pred):.2f}")
print(f"Model Coefficients: {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_test,y_test, color='blue', alpha=0.5, label='Actual Prices')
plt.plot(X_test,model.predict(X_test), color='red', linewidth=2, label='Regression Line')
plt.title('House Price Prediction using Linear Regression', fontsize=14)
plt.xlabel('Number of Rooms (RM)', fontsize=12)
plt.ylabel('Median Value in $1000s (MEDV)', fontsize=12)
plt.legend()
plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
plt.show()