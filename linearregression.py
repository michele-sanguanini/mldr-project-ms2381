import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'customerID' in cat_cols: cat_cols.remove('customerID')
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

num_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_train = jnp.array(X_train.values, dtype=jnp.float32)
X_test = jnp.array(X_test.values, dtype=jnp.float32)
y_train = jnp.array(y_train.values, dtype=jnp.float32)
y_test = jnp.array(y_test.values, dtype=jnp.float32)

print(X_train.shape, y_train.shape)
print(X_train[0], y_train[0])

params = {
    'w': jnp.zeros(X_train.shape[1:]),
    'b': 0.
}

def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']

def loss_fn(params, X, y):
    return jnp.mean(jnp.square(forward(params, X) - y))

optimizer = optax.sgd(learning_rate=0.05)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, X, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for _ in range(50):
    params, opt_state, loss = train_step(params, opt_state, X_train, y_train)
    print(loss)

# Compute validation loss
val_loss = loss_fn(params, X_test, y_test)
print('Validation loss:', val_loss)


def predict(params, X):
    return (forward(params, X) > 0.5).astype(jnp.float32)