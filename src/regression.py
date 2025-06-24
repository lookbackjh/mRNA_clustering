from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm    
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

            # 1. Y와 X 데이터 분리 (이전과 동일)
            y_index = feature_indices[0]
            x_indices = feature_indices[1 : 1 + self.args.num_nearest_points]

            Y = self.data_for_regression.iloc[:, y_index]
            X = self.data_for_regression.iloc[:, x_indices].copy()
            X['status'] = self.data_for_regression['status']

            # 2. 범주형 변수 처리 (이전과 동일)
            X = pd.get_dummies(X, columns=['status'], drop_first=True)

            # 3. [핵심] 선형 회귀 모델 학습 (statsmodels 사용)
            # statsmodels는 절편(intercept)을 자동으로 추가하지 않으므로, 수동으로 상수항을 추가해야 합니다.
            X_with_const = sm.add_constant(X)
            
            # OLS 모델을 정의하고 학습시킵니다.
            model = sm.OLS(Y, X_with_const)
            results = model.fit()

            # 4. [핵심] 결과 계산 및 저장 (statsmodels 결과 객체 사용)
            pvalues = results.pvalues
            r2_score = results.rsquared
            coefficients = results.params # params가 계수와 절편을 모두 포함
            intercept = results.params['const'] # 'const' 키로 절편 접근
            predictions = results.predict(X_with_const)

            # 학습된 모델과 결과를 딕셔너리에 저장
            self.regression_models[cluster_id] = results # results 객체 자체를 저장
            self.regression_results[cluster_id] = {
                'r2_score': r2_score,
                'coefficients': coefficients,
                'pvalues': pvalues,
                'intercept': intercept,
                'predictions': predictions
            }

            # 5. [핵심] 결과 요약 출력 (summary() 함수 사용)
            print(f"--- 클러스터 {cluster_id} 분석 결과 요약 ---")
            # summary() 함수는 R-squared, 계수, 표준오차, t-통계량, p-value 등 모든 정보를 담고 있습니다.
            print(results.summary())
            print("-" * 80)

        print("\n모든 클러스터에 대한 회귀 분석이 완료되었습니다.")

