import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'customerID' in cat_cols: cat_cols.remove('customerID')
for col in cat_cols:
    df[col] = df[col].astype('category')

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

num_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


model = xgb.XGBClassifier(
    tree_method='hist',
    enable_categorical=True,
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=50,
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)