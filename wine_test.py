import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


# model to predict wine quality 

df = pd.read_csv('winequality-red.csv', sep=';')
info = df.describe()
target = df.quality
X = df.drop('quality', axis=1)
print(np.unique(target))
trainX, testX, trainY, testY = train_test_split(X, target, test_size=0.2,
                                              random_state=123, stratify=target)

# declare the scaler for standardizing the train and test data
scalerv = preprocessing.StandardScaler().fit(trainX)
# scale the scale data
trainX_scaled = scalerv.transform(trainX)
testX_scaled = scalerv.transform(testX)

# create our model cross validation pipeline
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

# create the hyperparameters to test for 
hyperparams = {
            'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
            'randomforestregressor__max_depth': [None, 5, 3, 1]
        }


clf = GridSearchCV(pipeline, hyperparams, cv=10)
clf.fit(trainX, trainY)

# print out the best hyperparameters from the CV test
print(clf.best_params_)

# using the best hyperparameters predict for the test set
predY = clf.predict(testX)

# measure the level of accuracy
acc = r2_score(testY, predY)

print(mean_squared_error(testY, predY))

# save the model to a pickle file
joblib.dump(clf, 'rf_regressor.pkl')

# to load the model
# clf = joblib.load('rf_regressor.pkl')

