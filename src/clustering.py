from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import spearmanr   
class Clustering:
    def __init__(self,args, data):
        self.args = args
        self.data = data

    def calculate_distance_matrix(self):
        """
        Calculate the distance matrix for the data.
        """
        # Assuming self.data is a DataFrame with shape (n_samples, n_features)
        # Convert DataFrame to NumPy array for distance calculation
        data_array = self.data.values
        # Calculate the distance matrix using Euclidean distance
        if self.args.distance_metric == 'euclidean':
            # Use pdist to calculate pairwise distances and then convert to square form
            distance_matrix = distance.pdist(data_array, metric='euclidean')
            self.distance_matrix = distance.squareform(distance_matrix)
       
        elif self.args.distance_metric == 'pearson':
            # since pdist does not support 'spearman', we need to use a different approach
            # Calculate the pairwise Spearman correlation
            corr_matrix = self.data.T.corr(method='pearson')
            # Convert correlation to distance
            self.distance_matrix = 1 - corr_matrix

        elif self.args.distance_metric == 'spearman':
            # Calculate the pairwise Spearman correlation

            corr_matrix = self.data.T.corr(method='spearman')
            # Convert correlation to distance
            self.distance_matrix = 1 - corr_matrix
        else:
            raise ValueError(f"Unsupported distance metric: {self.args.distance_metric}")
    
    def low_dimensional_representation(self):

        #using distance matrix, convert 249*249 into 249*embedding_num
        # iuse pca
        if self.args.embedding_method =='pca':

            pca = PCA(n_components=self.args.embedding_num)
            # Fit PCA on the distance matrix
            pca.fit(self.distance_matrix)
            # Transform the data to low-dimensional representation
            low_dimensional_data = pca.transform(self.distance_matrix)
            # Convert the low-dimensional data back to a DataFrame
            self.low_dimensional_df = pd.DataFrame(low_dimensional_data, columns=[f'PC{i+1}' for i in range(self.args.embedding_num)])
            # Add the original index as a column
            print("Low-dimensional representation shape:", self.low_dimensional_df)

        elif self.args.embedding_method == 'laplacian':
            # 1. 거리 행렬을 유사도 행렬로 변환합니다.
            # sigma 값은 데이터 스케일에 따라 조정해야 하는 중요한 하이퍼파라미터입니다.
            # 거리의 평균값 등을 기준으로 설정해 볼 수 있습니다.
            sigma = self.args.sigma if hasattr(self.args, 'sigma') else 1.0 
            affinity_matrix = np.exp(-self.distance_matrix ** 2 / (2. * sigma ** 2)) #rbf kernel
            
            embedding = SpectralEmbedding(n_components=self.args.embedding_num, 
                                        affinity='precomputed', # 'precomputed'도 가능하지만, 이쪽이 더 안정적일 수 있음
                                        random_state=42)
            low_dimensional_data = embedding.fit_transform(affinity_matrix)
            self.low_dimensional_df = pd.DataFrame(low_dimensional_data, columns=[f'Dim{i+1}' for i in range(self.args.embedding_num)])
        else:
            raise ValueError(f"Unsupported embedding method: {self.args.embedding_method}")



        return 

    def optimal_cluster_num_check(self, max_clusters=10):
        # I just want you to check the drawing of the elbow method and silhouette score
        silhouette_scores = []
        inertia = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            inertia.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(self.data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        # Plotting the inertia

        plt.figure(figsize=(12, 6))
        plt.plot(range(2, max_clusters + 1), inertia, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()
        # save the inertia plot
        plt.savefig('figures/elbow_method.png')
        # Determine the optimal number of clusters based on silhouette score
        

    def kmeans(self, n_clusters=3):
        # data is in shape of n_features x n_samples, I want to do clustering based on the features. (possibley 249*29)

        distance_matrix = self.calculate_distance_matrix()
        self.low_dimensional_representation()  # Get low-dimensional representation

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)

        # 클러스터링에 사용할 수치형 데이터
        X = self.low_dimensional_df

        self.data['cluster'] = kmeans.fit_predict(X)
        self.cluster_centers = np.array(kmeans.cluster_centers_)

        print(self.cluster_centers[0])

        # --- [요청사항 2 ] 각 클러스터 중심에 가장 가까운 샘플 10개를 딕셔너리로 저장 ---
        print("\n--- [2] 각 클러스터 중심에 가장 가까운 샘플 10개 (딕셔너리 저장) ---")

        # 결과를 저장할 빈 딕셔너리를 생성합니다.
        self.nearest_samples_dict = {}

        # 모든 데이터 포인트와 클러스터 중심점 간의 유클리드 거리를 계산합니다.
        dists = distance.cdist(self.low_dimensional_df.iloc[:,:], self.cluster_centers)

        # 각 클러스터에 대해 반복합니다.
        for i in range(n_clusters):
            # i번 클러스터에 대한 거리 정보만 추출합니다.
            distances_to_center_i = dists[:, i]
            
            # 거리가 가까운 순서대로 샘플의 인덱스를 정렬합니다.
            nearest_sample_indices = np.argsort(distances_to_center_i)
            
            # 가장 가까운 10개의 샘플 인덱스를 가져옵니다.
            top_nearest_indices = nearest_sample_indices[:self.args.num_nearest_points]
            
            # [핵심] 딕셔너리에 {클러스터 번호: 인덱스 리스트} 형태로 저장합니다.
            # .tolist()를 사용하여 NumPy 배열을 일반 파이썬 리스트로 변환합니다.
            self.nearest_samples_dict[i] = top_nearest_indices.tolist()

        # 최종적으로 생성된 딕셔너리를 출력하여 확인합니다.
        print("최종 저장된 딕셔너리:")
        print(self.nearest_samples_dict)
        return self.data

    # def hierarchical(self, n_clusters=3):
    #     from sklearn.cluster import AgglomerativeClustering
    #     clustering = AgglomerativeClustering(n_clusters=n_clusters)
    #     self.data['cluster'] = clustering.fit_predict(self.data)
    #     return self.data