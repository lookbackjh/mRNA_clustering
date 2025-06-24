from sklearn.linear_model import LinearRegression
import pandas as pd
class Regression:
    def __init__(self, args, clustered_data, clustering_centers): 
        self.args = args
        self.clustered_data = clustered_data
        self.clustering_centers = clustering_centers

    def process_for_regression(self):
        # I want to make the columns vertically,  for example, for first 10 column the next 10 columns should be at the end of the first 10 columns and so on.
        # the data i in shape of n_features * n_samples, and the samples consists of 3 categories. 
        # what I want to do is to make data in to shape of n_features * n_samples * n_categories 
        data_normal=self.clustered_data.iloc[:, :10]
        data_abnormal=self.clustered_data.iloc[:, 10:19]
        data_alzheimer=self.clustered_data.iloc[:, 19:29]
        # Concatenate the data vertically

        data_normal=data_normal.T
        data_abnormal=data_abnormal.T
        data_alzheimer=data_alzheimer.T
        # now concatenate the data
        self.data_for_regression = pd.concat([data_normal, data_abnormal, data_alzheimer], axis=0)
        # first 10 columns are normal, next 9 columns are abnormal and the last 10 columns are alzheimer.
        self.data_for_regression['status'] = ['normal'] * data_normal.shape[0] + ['abnormal'] * data_abnormal.shape[0] + ['alzheimer'] * data_alzheimer.shape[0]


        
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)