from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'Data'

df = pd.read_csv(DATA_DIR / '4) house Prediction Data Set.csv', sep=r'\s+', engine='python', header=None)

median_price = df.iloc[:, 13].median()

df['is_expensive'] = (df.iloc[:, 13] > median_price).astype(int)

X = df.iloc[:, [5]]
y = df['is_expensive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print('---Classification Report---')
print(classification_report(y_test,y_pred))
print(f'Accuracy Score: {accuracy_score(y_test,y_pred):.4f}')
print(f'Precision Score: {precision_score(y_test,y_pred):.4f}')
print(f'Recall Score: {recall_score(y_test,y_pred):.4f}')
print(f'F1 Score: {f1_score(y_test,y_pred):.4f}')