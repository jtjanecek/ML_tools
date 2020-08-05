# Import libraries
import pandas as pd
from sklearn.linear_model import Lasso

# Read in the data
df = pd.read_csv('data.csv')

# Extract X, y
y = df['outcome'].values
df = df.drop(['outcome'], axis=1)
X = df.values
labels = df.columns

model = Lasso(alpha=.01)
model.fit(X, y)

for i in range(len(model.coef_)):
	print(labels[i], model.coef_[i])
