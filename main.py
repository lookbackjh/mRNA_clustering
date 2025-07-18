from src.preprocess import Preprocess
from src.clustering import Clustering
from src.feature_selector import FDRFeatureSelector
from src.regression import Regression

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Process Excel data.")
    parser.add_argument('--data_path', type=str, default='data2.xlsx', help='Path to the Excel file')
    parser.add_argument('--small_value', type=float, default=1e-6, help='Small value to replace zeros in the data')
    parser.add_argument('--num_clusters', type=int, default=4, help='Number of clusters for clustering algorithms')
    parser.add_argument('--num_nearest_points', type=int, default=11, help='Number of nearest points to consider for each cluster center')
    parser.add_argument('--normalization_method', type=str, default='log_naive', help='Normalization method to apply to the data, options: log_naive, clr, prob, naive')
    parser.add_argument('--fdr_threshold', type=float, default=0.1, help='FDR threshold for feature selection')
    parser.add_argument('--distance_metric', type=str, default='pearson', help='euclidean, spearman, pearson, cosine, or correlation distance metric')
    parser.add_argument('--embedding_num', type=int, default=10, help='Number of dimensions for low-dimensional representation')
    parser.add_argument('--embedding_method', type=str, default='laplacian', help='Method for low-dimensional representation, options: pca, laplacian')
    parser.add_argument('--count_threshold', type=float, default=4, help='Threshold for count-based regression')
    parser.add_argument('--to_split', type=bool, default=True, help='Whether to split the data ')
    parser.add_argument('--num_features_to_display', type=int, default=20, help='Number of features to display after FDR correction')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha value for lasso regression')
    parser.add_argument('--group_alpha', type=float, default=0.1, help='Alpha value for group lasso regression')
    parser.add_argument('--do_grouplasso', type=bool, default=True, help='boolean to decide whether to do group lasso or not')
    parser.add_argument('--reference_variables', nargs='*', default=['hsa-miR-103a-3p', 'hsa-miR-25-3p'], help='List of reference variables (feature names) to be unpenalized in group lasso.') #기본참여변수설정
    parser.add_argument('--target_features', nargs='*', default=['hsa-mir-temp_1_1', 'hsa-mir-temp_3_1', 'hsa-mir-temp_4_1', 'hsa-mir-temp_5_1'], help='List of feature names to be used as target variables for regression. If empty, cluster centers will be used.') # 'hsa-mir-temp_1_1', 'hsa-mir-temp_3_1', 'hsa-mir-temp_4_1', 'hsa-mir-temp_5_1'
    parser.add_argument('--num_correlated_features', type=int, default=20, help='Number of top correlated features to select as explanatory variables when target_features are specified.')
    #parser.add_argument('--num_features_to_display', type=int, default=20, help='Number of features to display  correction')

    return parser.parse_args()
    

def main():
    args = parse_args() 
    # Load data from Excel file
    preprocessor = Preprocess(args)
    data = preprocessor.get_processed_data() # data would be shape of n_features x n_samples

    fdr_feature_selector = FDRFeatureSelector(args, data,preprocessor.feature_dict)  # here, preprocessor.feature_dict is a dictionary that maps feature indices to feature names
    fdr_feature_selector.select_features()  # Select features based on FDR
    fdr_feature_selector.display_sorted_features()  # Display sorted features by p-value

    #print(f"Debug: preprocessor.feature_dict: {preprocessor.feature_dict}")

    clustering = Clustering(args, data, preprocessor.feature_dict)
    # clustering.optimal_cluster_num_check()  # Check for optimal number of clusters

    # target_features가 있으면 해당 피처를 중심으로 클러스터링, 없으면 일반 KMeans
    if args.target_features:
        clustering.kmeans(target_features=args.target_features)
    else:
        clustering.kmeans(n_clusters=args.num_clusters)

    regression = Regression(args, clustering, preprocessor.feature_dict, args.reference_variables, args.target_features)
    regression.do_lasso_regression()  # Process data for regression
    #regression._get_data_for_lasso() # Perform lasso regression on the clustered data


if __name__ == "__main__":
    main()