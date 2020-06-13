import numpy as np 
import pandas as pd
from matplotlib.pylab import rcParams
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

rcParams['figure.figsize']= 12, 10

# define input array eith angles from 60 to 300 deg converting to radians
x = np.array([i*np.pi/180 for i in range(60, 300, 4)])
np.random.seed(10)
y = np.sin(x) + np.random.normal(0, 0.15, len(x))
data = pd.DataFrame(np.column_stack([x, y]), columns=['x', 'y'])

for i in range(2, 10):
  column = 'x_%d'%i
  data[column] = data.x**i

X = data.drop('y', axis=1)
y = data.y

alph = 0.01
# training using rigde regression
ridge_reg = Ridge(alpha=alph, normalize=True)
ridge_reg.fit(X, y)
pred1 = ridge_reg.predict(X)
score1 = ridge_reg.score(X, y)
acc1 = mean_squared_error(y, pred1)
print("Score: %f"%score1)
print(" error: %f"%acc1)
print("===========")

# training using lasso reg
lasso_reg = Lasso(alpha=alph, normalize=True)
lasso_reg.fit(X, y)
pred2 = lasso_reg.predict(X)
score2 = lasso_reg.score(X, y)
acc2 = mean_squared_error(y, pred2)
print("Score: %f"%score2)
print(" error: %f"%acc2)
print("===========")
# training using elastic net regression 
elastic_net = ElasticNet(alpha=alph, normalize=True)
elastic_net.fit(X, y)
pred3 = elastic_net.predict(X)
score3 = elastic_net.score(X, y)
acc3 = mean_squared_error(y, pred3)
print("Score: %f"%score3)
print(" error: %f"%acc3)
print("===========")