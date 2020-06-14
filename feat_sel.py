from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
#from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def select_features(data_x, data_y=None, fs=None, score_function=None, num_feat=10):
    # Array for selected features
    sel_features = []
    if fs is None:
        assert(data_y is not None), "data_y required when fs is None"
        assert(score_function is not None), "score_function required when fs is None"
        # init the feature selector model
        feature_selector = SelectKBest(score_func=score_function, k=num_feat)
        feature_selector.fit(data_x, np.ravel(data_y))
    else:
        feature_selector = fs
        
    # append the selected features
    for boole, feature in zip(feature_selector.get_support(), data_x.columns.values):
        if boole:
            sel_features.append(feature)
            
    data_x = feature_selector.transform(data_x)
    data_x = pd.DataFrame(data_x, columns=sel_features)
    return data_x, feature_selector
    


if __name__ == "__main__":    
    
    # make the regression datasets
    X, y = make_regression(n_samples=1000, n_features=50, n_informative=10, random_state=1, noise=0.5)
    X = pd.DataFrame(X, columns=['feature_%d'%i for i in range(50)])
    y = pd.DataFrame(y)
    # split into train and validation set
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
    train_Xt, fs = select_features(train_X, train_y, score_function=f_regression)
    
    val_Xt, fs = select_features(val_X, fs=fs)
    
    for i in range(len(fs.scores_)):
        print("feature %d: %f"%(i, fs.scores_[i]))
    
    plt.bar(["feat_%d"%i for i in range(len(fs.scores_))], fs.scores_)
    plt.xticks(rotation=-90)
    plt.show()
    
