from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from scipy.spatial import distance
class Clustering:
    def __init__(self,args, data):
        self.args = args
        self.data = data

    def optimal_cluster_num_check(self, max_clusters=10):

        import numpy as np
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
        import matplotlib.pyplot as plt
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
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)

        # 클러스터링에 사용할 수치형 데이터
        X = self.data

        self.data['cluster'] = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_

        # --- [요청사항 1은 이전과 동일] ---
        # (이해를 돕기 위해 생략)
        feature_names = X.columns

        for i, center in enumerate(self.cluster_centers):
            # 중심점에서 값이 가장 큰 feature의 인덱스를 찾습니다.
            most_important_feature_index = np.argmax(center)
            # 해당 인덱스의 feature 이름과 값을 가져옵니다.
            most_important_feature_name = feature_names[most_important_feature_index]
            most_important_feature_value = center[most_important_feature_index]

            print(f"\n[클러스터 번호 {i}]")
            print(f"  > 가장 중요한 Feature: '{most_important_feature_name}' (인덱스: {most_important_feature_index})")
            print(f"  > 해당 Feature의 중심점 값: {most_important_feature_value:.4f}")

            # (추가) 상위 5개 feature를 보고 싶을 경우
            top_5_feature_indices = np.argsort(center)[-5:][::-1] # 값이 큰 순서대로 5개 인덱스
            top_5_features = feature_names[top_5_feature_indices]
            print(f"  > 상위 5개 중요 Feature: {top_5_features.tolist()}")


        # --- [요청사항 2 수정] 각 클러스터 중심에 가장 가까운 샘플 10개를 딕셔너리로 저장 ---
        print("\n--- [요청사항 2] 각 클러스터 중심에 가장 가까운 샘플 10개 (딕셔너리 저장) ---")

        # 결과를 저장할 빈 딕셔너리를 생성합니다.
        self.nearest_samples_dict = {}

        # 모든 데이터 포인트와 클러스터 중심점 간의 유클리드 거리를 계산합니다.
        dists = distance.cdist(self.data.iloc[:,:29], self.cluster_centers)

        # 각 클러스터에 대해 반복합니다.
        for i in range(n_clusters):
            # i번 클러스터에 대한 거리 정보만 추출합니다.
            distances_to_center_i = dists[:, i]
            
            # 거리가 가까운 순서대로 샘플의 인덱스를 정렬합니다.
            nearest_sample_indices = np.argsort(distances_to_center_i)
            
            # 가장 가까운 10개의 샘플 인덱스를 가져옵니다.
            top_10_nearest_indices = nearest_sample_indices[:10]
            
            # [핵심] 딕셔너리에 {클러스터 번호: 인덱스 리스트} 형태로 저장합니다.
            # .tolist()를 사용하여 NumPy 배열을 일반 파이썬 리스트로 변환합니다.
            self.nearest_samples_dict[i] = top_10_nearest_indices.tolist()

        # 최종적으로 생성된 딕셔너리를 출력하여 확인합니다.
        print("최종 저장된 딕셔너리:")
        print(self.nearest_samples_dict)
        return self.data

    def hierarchical(self, n_clusters=3):
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        self.data['cluster'] = clustering.fit_predict(self.data)
        return self.data