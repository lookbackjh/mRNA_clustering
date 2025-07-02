from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import Lasso
import statsmodels.api as sm  
import numpy as np
from sklearn.preprocessing import StandardScaler
from group_lasso import GroupLasso
class Regression:
    def __init__(self, args, clustering, feature_dict=None): 
        self.args = args
        self.clustered_data = clustering.data # Get the clustered data from the clustering object
        self.clustering_centers = clustering.cluster_centers # Get the clustering centers
        self.c_nearest_points = clustering.nearest_samples_dict  # Get the nearest points for each cluster
        self.feature_dict = feature_dict if feature_dict is not None else {}
        #self.lasso_data=None


    def _tranform_data_for_lasso(self,X):
        """
        """
        # for every column in X I want to  split into 6 columns col_normal_zero, col_normal_nonzero, col_abnormal_zero, col_abnormal_nonzero, col_alzheimer_zero, col_alzheimer_nonzero

        colnames=[]
        group_indicators = [] # this should be a list containing the group indicators for each column, which will be used in group lasso regression.
        for idx,col in enumerate(X.columns):
            col_normal_zero = f"{col}_N_zero"
            col_normal_nonzero = f"{col}_N_nonzero"
            col_abnormal_zero = f"{col}_M_zero"
            col_abnormal_nonzero = f"{col}_M_nonzero"
            col_alzheimer_zero = f"{col}_A_zero"
            col_alzheimer_nonzero = f"{col}_A_nonzero"

            colnames.extend([col_normal_zero, col_normal_nonzero, col_abnormal_zero, col_abnormal_nonzero, col_alzheimer_zero, col_alzheimer_nonzero])
            # I want to add the group indicators for each column
            group_indicators.extend([idx]*6)
            
        # Fill the columns based on the status
        transformed_data = pd.DataFrame(index=X.index, columns=colnames)

        # now lets fill in the columns from original data, lets see column vectors. 
        for col in X.columns:
            #get column vector
            col_vector = X[col]
            # I'd use two boolean masks to identify zero and non-zero values for each status # till index 10 is normal, 10 to 19 is abnormal, and 19 to 29 is alzheimer
            indices = np.arange(29)

            # 각 조건에 맞는 마스크 생성
            normal_mask = indices < 10
            abnormal_mask = (indices >= 10) & (indices < 19)
            alzheimer_mask = (indices >= 19) & (indices < 29)
            # and zero mask and non-zero mask are masks for splitting small values. 
            zeromask = col_vector <self.args.count_threshold
            nonzeromask = col_vector >= self.args.count_threshold
            # Fill the transformed_data DataFrame based on these masks
            col_vector_normal_zero = col_vector.where(normal_mask & zeromask, 0)
            col_vector_normal_nonzero = col_vector.where(normal_mask & nonzeromask, 0)
            col_vector_abnormal_zero = col_vector.where(abnormal_mask & zeromask,
                                                            0)  
            col_vector_abnormal_nonzero = col_vector.where(abnormal_mask & nonzeromask, 0)
            col_vector_alzheimer_zero = col_vector.where(alzheimer_mask & zeromask,
                                                            0)
            col_vector_alzheimer_nonzero = col_vector.where(alzheimer_mask & nonzeromask, 0)

            col_normal_zero = f"{col}_N_zero"
            col_normal_nonzero = f"{col}_N_nonzero"
            col_abnormal_zero = f"{col}_M_zero"
            col_abnormal_nonzero = f"{col}_M_nonzero"
            col_alzheimer_zero = f"{col}_A_zero"
            col_alzheimer_nonzero = f"{col}_A_nonzero"

            # Fill the transformed_data DataFrame
            transformed_data[col_normal_zero] = col_vector_normal_zero
            transformed_data[col_normal_nonzero] = col_vector_normal_nonzero
            transformed_data[col_abnormal_zero] = col_vector_abnormal_zero
            transformed_data[col_abnormal_nonzero] = col_vector_abnormal_nonzero
            transformed_data[col_alzheimer_zero] = col_vector_alzheimer_zero
            transformed_data[col_alzheimer_nonzero] = col_vector_alzheimer_nonzero 
        # Fill NaN values with 0


        return transformed_data, group_indicators
        

    def _get_data_for_regression(self):
        """
        Returns the data prepared for regression.
        """

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

        return self.data_for_regression

    
    def do_lasso_regression(self):
        """
        Perform lasso regression on the clustered data.
        This method is a placeholder and needs to be implemented.
        
        """
        self.data_for_regression = self._get_data_for_regression()  
        for cluster_id, feature_indices in self.c_nearest_points.items():
            print(f"\n>>>클러스터 {cluster_id}에 대한 Lasso 회귀 분석 시작...")

            # 1. Y와 X 데이터 분리 (이전과 동일)
            y_index = feature_indices[0]
            x_indices = feature_indices[1 : 1 + self.args.num_nearest_points]

            Y = self.data_for_regression.iloc[:, y_index]
            X = self.data_for_regression.iloc[:, x_indices].copy()
            X_transformed,group_indicators = self._tranform_data_for_lasso(X)  # Transform the data for lasso regression

            scaler= StandardScaler()
            # Scale the data

            # this is general lasso
            X_transformed_scaled = scaler.fit_transform(X_transformed)
  

            # I also want to do grouplasso       
            # If do_grouplasso is True, perform group lasso regression
            if  self.args.do_grouplasso: 
                print(f"--- 클러스터 {cluster_id} Group Lasso 회귀 분석 시작 ---")
                group_lasso = GroupLasso(groups=group_indicators, group_reg=self.args.group_alpha, l1_reg=self.args.alpha,supress_warning=True)
                group_lasso.fit(X_transformed_scaled, Y)
                group_lasso_coef = group_lasso.coef_
                print(f"Group Lasso Coefficients: {group_lasso_coef}")
            else:
                print(f"--- 클러스터 {cluster_id} Lasso 회귀 분석 시작 ---")
                lasso = Lasso(alpha=self.args.alpha, max_iter=1000)
                lasso.fit(X_transformed_scaled, Y)
                lasso_coef = lasso.coef_
                print(f"Lasso Coefficients: {lasso_coef}")

        pass



    
    def _split_data(self):
        pass


    def run_regression_analysis(self):
        # I want to make the columns vertically,  for example, for first 10 column the next 10 columns should be at the end of the first 10 columns and so on.
        # the data i in shape of n_features * n_samples, and the samples consists of 3 categories. 
        # what I want to do is to make data in to shape of n_features * n_samples * n_categories 

        # Now I want to do regression based on the clustering centers.
        #self.model = LinearRegression()
        self.data_for_regression = self._get_data_for_regression()  # 데이터 준비

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
            # define indexes as row numbers

            X_with_const.index = range(len(X_with_const))  # 인덱스를 0부터 시작하는 정수로 설정
            Y.index = range(len(Y))  # Y의 인덱스도 동일하게
            # data type as float
            X_with_const = X_with_const.astype(float)
            Y = Y.astype(float)


            # I want to change the coluumn nmae of X_with_const if the index of feature is in the feature_dict
            X_with_const.rename(columns={i: self.feature_dict.get(i, f'feature_{i}') for i in X_with_const.columns}, inplace=True)

            model = sm.OLS(Y, X_with_const)
            results = model.fit()

            # 4. [핵심] 결과 계산 및 저장 (statsmodels 결과 객체 사용)
            pvalues = results.pvalues
            r2_score = results.rsquared
            coefficients = results.params # params가 계수와 절편을 모두 포함
            intercept = results.params['feature_const'] # 'const' 키로 절편 접근
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