# HR-Analytics
This project predicts whether an employee will leave the company using the IBM HR Analytics Dataset. The model is built using the Random Forest Classifier in Python.

Dataset
Source: IBM HR Analytics Dataset
Features: Employee demographics, job role, work experience, and satisfaction levels.
Target Variable: Attrition (Yes = Employee left, No = Employee stayed).
Tech Stack
🔹 Python (Pandas, NumPy, Matplotlib, Seaborn)
🔹 Machine Learning (Scikit-Learn, Random Forest)
🔹 Google Colab / Jupyter Notebook

Model & Accuracy
Model: RandomForestClassifier(n_estimators=100, random_state=42)
Accuracy: 87.75%
Issue: Class imbalance – the model predicts non-leaving employees well but struggles with those who leave.
Implementation
1. Install Dependencies
sh
Copy
Edit
pip install pandas numpy scikit-learn imbalanced-learn
2. Load Data
python
Copy
Edit
import pandas as pd
df = pd.read_csv("/content/HR-Employee-Attrition.csv")
3. Preprocess Data & Train Model
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables  
X = df.drop(columns=["Attrition_Yes"])  
y = df["Attrition_Yes"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Results
Metric	Precision	Recall	F1-Score
Stayed (False)	0.88	1.00	0.93
Left (True)	0.80	0.10	0.18
📉 Issue: The model struggles with predicting employees who leave.
