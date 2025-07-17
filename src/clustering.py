from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.manifold import MDS

class Clustering:
    def __init__(self, args, data, feature_dict=None):
        """
        객체 생성 시에는 입력값만 저장하고, 실제 계산은 하지 않습니다.
        결과를 캐싱할 비공개(private) 속성들을 None으로 초기화합니다.
        """
        self.args = args
        self.data = data.copy() # 원본 데이터 수정을 방지하기 위해 복사본 사용
        self.feature_dict = feature_dict if feature_dict is not None else {}
        
        # ★ 1. 결과물을 캐싱하기 위한 '비공개' 속성들을 선언합니다.
        self._distance_matrix = None
        self._low_dimensional_df = None
        
        # kmeans의 결과물은 파라미터(n_clusters)에 따라 달라지므로 property로 만들지 않습니다.
        self.cluster_labels = None
        self.cluster_centers = None
        self.nearest_samples_dict = None

    @property
    def distance_matrix(self):
        """
        'distance_matrix' 속성에 처음 접근할 때만 실제 계산을 수행합니다.
        """
        # ★ 2. 캐시된 결과가 있는지 확인
        if self._distance_matrix is None:
            print("--- Computing Distance Matrix (this happens only once) ---")
            data_array = self.data.values
            
            if self.args.distance_metric == 'euclidean':
                dist_mat = distance.pdist(data_array, metric='euclidean')
                self._distance_matrix = distance.squareform(dist_mat)
            elif self.args.distance_metric in ['pearson', 'spearman']:
                corr_matrix = self.data.T.corr(method=self.args.distance_metric)
                self._distance_matrix = 1 - corr_matrix.values # DataFrame -> numpy array
            else:
                raise ValueError(f"Unsupported distance metric: {self.args.distance_metric}")
        
        # ★ 3. 캐시된 결과를 반환
        return self._distance_matrix

    @property
    def low_dimensional_df(self):
        """
        'low_dimensional_df' 속성에 처음 접근할 때만 실제 계산을 수행합니다.
        이 과정에서 self.distance_matrix가 필요하면, 해당 property가 자동으로 호출됩니다.
        """
        if self._low_dimensional_df is None:
            print("--- Computing Low-Dimensional Representation (this happens only once) ---")
            
            if self.args.embedding_method == 'pca':
                mds = MDS(n_components=self.args.embedding_num)
                # ★ 4. '연쇄 반응': self.distance_matrix에 접근하면 위 property가 실행됩니다.
                low_dim_data = mds.fit_transform(self.distance_matrix)
                self._low_dimensional_df = pd.DataFrame(low_dim_data, columns=[f'MDS{i+1}' for i in range(self.args.embedding_num)])

            elif self.args.embedding_method == 'laplacian':
                sigma = self.args.sigma if hasattr(self.args, 'sigma') else 1.0
                affinity_matrix = np.exp(-self.distance_matrix ** 2 / (2. * sigma ** 2))
                embedding = SpectralEmbedding(n_components=self.args.embedding_num, affinity='precomputed', random_state=42)
                low_dim_data = embedding.fit_transform(affinity_matrix)
                self._low_dimensional_df = pd.DataFrame(low_dim_data, columns=[f'Dim{i+1}' for i in range(self.args.embedding_num)])
            else:
                raise ValueError(f"Unsupported embedding method: {self.args.embedding_method}")

        return self._low_dimensional_df

    def optimal_cluster_num_check(self, max_clusters=10):
        """
        최적의 클러스터 수를 확인합니다. 
        내부에서 self.low_dimensional_df에 접근하는 순간 모든 계산이 자동으로 처리됩니다.
        """
        print("\n--- Checking for Optimal Number of Clusters ---")
        silhouette_scores = []
        inertia = []
        
        # ★ 5. 사용자는 그냥 속성을 호출하면 됩니다. 계산은 알아서 처리됩니다.
        embedding_data = self.low_dimensional_df
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding_data)
            inertia.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(embedding_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), inertia, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='r')
        plt.title('Silhouette Score')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def kmeans(self, n_clusters=3, target_features=None):
        """
        K-Means 클러스터링을 수행합니다.
        target_features가 제공되면, 해당 특성을 초기 클러스터 중심으로 사용합니다.
        """
        X = self.low_dimensional_df
        
        if target_features and self.feature_dict:
            print(f"\n--- Performing K-Means using specified target features as initial centers ---")
            # feature_dict의 역매핑을 생성합니다.
            feature_name_to_idx = {name: idx for idx, name in self.feature_dict.items()}
            
            # target_features 이름에 해당하는 인덱스를 찾습니다.
            initial_center_indices = [feature_name_to_idx[name] for name in target_features if name in feature_name_to_idx]
            
            if not initial_center_indices:
                raise ValueError("Target features not found in the data.")

            # 저차원 표현에서 해당 인덱스의 데이터를 초기 중심으로 설정합니다.
            initial_centers = X.iloc[initial_center_indices].values
            n_clusters = len(initial_centers)
            
            print(f"Using {n_clusters} target features as initial centers.")

            kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, random_state=42)

        else:
            print(f"\n--- Performing K-Means with {n_clusters} clusters (k-means++) ---")
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        
        self.cluster_labels = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_
        self.data['cluster'] = self.cluster_labels

        # 각 클러스터 중심에 가장 가까운 샘플 찾기
        self.nearest_samples_dict = {}
        dists = distance.cdist(X, self.cluster_centers)
        for i in range(n_clusters):
            # 클러스터 중심에서 가장 가까운 원본 데이터 포인트를 찾습니다.
            # cdist는 X의 각 포인트와 cluster_centers 간의 거리를 계산합니다.
            # argsort는 각 클러스터 중심(열)에 대해 가장 가까운 X 포인트(행)의 인덱스를 오름차순으로 정렬합니다.
            nearest_indices = np.argsort(dists[:, i])
            
            # num_nearest_points 만큼의 인덱스를 저장합니다.
            num_points = self.args.num_nearest_points if hasattr(self.args, 'num_nearest_points') else 10
            self.nearest_samples_dict[i] = nearest_indices[:num_points].tolist()
        
        print("Clustering complete. Results are stored in the object.")