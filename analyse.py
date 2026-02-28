import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier

data = pd.read_csv("data/data.csv")
data.gender = data.gender.eq('Male').mul(1)
to_drop = []
for col in data.columns:
    print(col)
    if data[col][0] in ('Yes','No'):
        data[col] = data[col].eq('Yes').mul(1)
        print("mapped",col)
    elif not pd.api.types.is_numeric_dtype(data[col].dtype):
        to_drop.append(col)
print("DROPPING",to_drop)
data = data.drop(to_drop, axis=1)
X = data.drop(['Churn'], axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8)
# print(X_train, X_test, y_train, y_test)

pct = Perceptron(tol=1e-3, random_state=0).fit(X_train, y_train)
print("Perceptron score:",pct.score(X_test, y_test))

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=2, random_state=0).fit(X_train, y_train)
print("GBC score:",gbc.score(X_test, y_test))

xgb = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=1, objective='binary:logistic')
xgb.fit(X_train, y_train)
print("XGB score", xgb.score(X_test,y_test))