from sklearn.linear_model import LinearRegression
import pandas as pd
class Regression:
    def __init__(self, args, clustered_data, clustering_centers,c_nearest_points): 
        self.args = args
        self.clustered_data = clustered_data
        self.clustering_centers = clustering_centers
        self.c_nearest_points = c_nearest_points

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

        # Now I want to do regression based on the clustering centers.
        #self.model = LinearRegression()

        self.regression_models = {}
        self.regression_results = {}

        # 딕셔너리의 모든 클러스터에 대해 반복
        for cluster_id, feature_indices in self.c_nearest_points.items():
            print(f"\n>>> 클러스터 {cluster_id}에 대한 회귀 분석 시작...")

            # 1. Y와 X 데이터 분리
            # 첫 번째 인덱스를 Y (종속 변수)로 사용
            y_index = feature_indices[0]
            # 나머지 인덱스를 X (독립 변수)로 사용
            x_indices = feature_indices[1 : 1 + self.args.num_nearest_points] # 설정한 개수만큼 슬라이싱

            Y = self.data_for_regression.iloc[:, y_index]
            X = self.data_for_regression.iloc[:, x_indices].copy() # SettingWithCopyWarning 방지를 위해 .copy() 사용
            
            # 'status' 열을 X에 추가
            X['status'] = self.data_for_regression['status']

            # 2. 범주형 변수(status) 처리 (One-Hot Encoding)
            X = pd.get_dummies(X, columns=['status'], drop_first=True)

            # 3. 선형 회귀 모델 학습
            model = LinearRegression()
            model.fit(X.values, Y.values)

            # 4. 결과 계산 및 저장
            predictions = model.predict(X.values)
            r2_score = model.score(X.values, Y.values) # R-squared 값
            coefficients = model.coef_
            intercept = model.intercept_

            # 학습된 모델과 결과를 딕셔너리에 저장
            self.regression_models[cluster_id] = model
            self.regression_results[cluster_id] = {
                'r2_score': r2_score,
                'coefficients': coefficients,
                'intercept': intercept,
                'predictions': predictions
            }

            # 5. 결과 요약 출력
            print(f"--- 클러스터 {cluster_id} 분석 결과 요약 ---")
            print(f"  - R^2 Score: {r2_score:.4f}")
            print(f"  - Intercept (절편): {intercept:.4f}")
            # 계수는 feature 이름과 함께 출력하면 더 보기 좋습니다.
            coef_summary = {name: round(coef, 4) for name, coef in zip(X.columns, coefficients)}
            print(f"  - Coefficients (계수): \n{pd.Series(coef_summary)}")
            print("-" * 40)




        # Prepare the features and target variable
    

