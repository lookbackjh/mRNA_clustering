import pandas as pd
import numpy as np
from src.preprocess import Preprocess
from src.clustering import Clustering
from src.feature_selector import FDRFeatureSelector

from src.regression import Regression
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Process Excel data.")
    parser.add_argument('--data_path', type=str, default='data/data2.xlsx', help='Path to the Excel file')
    parser.add_argument('--small_value', type=float, default=1e-6, help='Small value to replace zeros in the data')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters for clustering algorithms')
    parser.add_argument('--num_nearest_points', type=int, default=11, help='Number of nearest points to consider for each cluster center')
    parser.add_argument('--normalization_method', type=str, default='naive', help='Normalization method to apply to the data, options: log, clr, naive')
    parser.add_argument('--fdr_threshold', type=float, default=0.1, help='FDR threshold for feature selection')
    parser.add_argument('--distance_metric', type=str, default='pearson', help='euclidean, spearman, pearson, cosine, or correlation distance metric')
    parser.add_argument('--embedding_num', type=int, default=10, help='Number of dimensions for low-dimensional representation')
    parser.add_argument('--embedding_method', type=str, default='pca', help='Method for low-dimensional representation, options: pca, laplacian')
    parser.add_argument('--count_threshold', type=float, default=0.1, help='Threshold for count-based regression')

    return parser.parse_args()
    

def main():
    args = parse_args() 
    # Load data from Excel file
    preprocessor = Preprocess(args)
    data = preprocessor.get_processed_data() # data would be shape of n_features x n_samples

    fdr_feature_selector = FDRFeatureSelector(args, data,preprocessor.feature_dict)
    fdr_feature_selector.select_features()  # Select features based on FDR
    fdr_feature_selector.display_sorted_features()  # Display sorted features by p-value

    clustering= Clustering(args, data)
    clustering.kmeans(n_clusters=args.num_clusters)  # Perform KMeans clustering

    regression = Regression(args,clustering)
    regression.do_regression()  # Process data for regression


if __name__ == "__main__":
    main()