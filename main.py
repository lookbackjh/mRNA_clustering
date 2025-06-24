import pandas as pd
import numpy as np
from src.preprocess import Preprocess
from src.clustering import Clustering

from src.regression import Regression
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Process Excel data.")
    parser.add_argument('--data_path', type=str, default='data/data2.xlsx', help='Path to the Excel file')
    parser.add_argument('--small_value', type=float, default=1e-6, help='Small value to replace zeros in the data')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters for clustering algorithms')
    parser.add_argument('--num_nearest_points', type=int, default=11, help='Number of nearest points to consider for each cluster center')
    return parser.parse_args()


def main():
    args = parse_args() 
    # Load data from Excel file
    preprocessor = Preprocess(args)
    data = preprocessor.get_processed_data() # data would be shape of n_features x n_samples
    print("Processed Data:")
    print(data)
    clustering= Clustering(args, data)
    # Check optimal number of clusters
    #clustering.optimal_cluster_num_check(max_clusters=20)
    c_result_data=clustering.kmeans(n_clusters=args.num_clusters)
    c_centers = clustering.cluster_centers # get the clustering centers.  
    # Please note that the index of the c_centers corresponds to the cluster that the data(center) belongs to.
    c_nearest_points=clustering.nearest_samples_dict
    print("Clustering Result:")
    print(c_nearest_points)


    # Now I want to do Regression based on the clustering centerings. 
    regression = Regression(args, c_result_data, c_centers,c_nearest_points)
    regression.do_regression()  # Process data for regression


if __name__ == "__main__":
    main()