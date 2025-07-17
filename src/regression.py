from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from group_lasso import GroupLasso

class Regression:
    """
    클러스터링 결과를 기반으로 회귀 분석(OLS, Group Lasso)을 수행하는 클래스.
    """
    def __init__(self, args, clustering, feature_dict=None):
        self.args = args
        self.clustered_data = clustering.data
        self.c_nearest_points = clustering.nearest_samples_dict
        self.feature_dict = feature_dict if feature_dict is not None else {}
        self.data_for_regression = None # 분석 전 데이터 준비를 위해 None으로 초기화

    def _get_data_for_regression(self):
        """
        회귀 분석에 사용될 데이터를 준비하고 Transpose하여 반환합니다.

        최적화: 여러 DataFrame을 각각 Transpose하지 않고, 원본을 한 번만 Transpose하여 효율성을 높였습니다.
        """
        # clustered_data를 먼저 Transpose하여 샘플을 행, 특성을 열로 만듭니다.
        transposed_data = self.clustered_data.T

        # 미리 정의된 인덱스를 기반으로 데이터를 분리합니다.
        data_normal = transposed_data.iloc[:10, :]
        data_abnormal = transposed_data.iloc[10:19, :]
        data_alzheimer = transposed_data.iloc[19:29, :]

        # status 열을 추가하며 데이터를 수직으로 결합합니다.
        data_normal['status'] = 'normal'
        data_abnormal['status'] = 'abnormal'
        data_alzheimer['status'] = 'alzheimer'

        # 최종 데이터를 생성하고 클래스 속성으로 저장합니다.
        self.data_for_regression = pd.concat([data_normal, data_abnormal, data_alzheimer], axis=0)
        return self.data_for_regression

    def _tranform_data_for_lasso(self, X):
        """
        Group Lasso 분석을 위해 원본 설명 변수(X)를 6개의 하위 특성으로 분해합니다.

        최적화: 반복문 내에서 DataFrame을 계속 수정하는 대신,
                 리스트에 작은 DataFrame들을 담아 마지막에 한 번만 결합(concat)하여 성능을 개선했습니다.
        """
        transformed_dfs = []
        group_indicators = []
        
        # 반복문 밖에서 고정된 마스크를 미리 생성합니다.
        indices = np.arange(len(X))
        normal_mask = indices < 10
        abnormal_mask = (indices >= 10) & (indices < 19)
        alzheimer_mask = indices >= 19

        for idx, col in enumerate(X.columns):
            col_vector = X[col]
            zeromask = col_vector < self.args.count_threshold
            nonzeromask = ~zeromask  # Boolean not 연산으로 더 간결하게 표현

            temp_data = {
                f"{col}_N_zero": col_vector.where(normal_mask & zeromask, 0),
                f"{col}_N_nonzero": col_vector.where(normal_mask & nonzeromask, 0),
                f"{col}_M_zero": col_vector.where(abnormal_mask & zeromask, 0),
                f"{col}_M_nonzero": col_vector.where(abnormal_mask & nonzeromask, 0),
                f"{col}_A_zero": col_vector.where(alzheimer_mask & zeromask, 0),
                f"{col}_A_nonzero": col_vector.where(alzheimer_mask & nonzeromask, 0),
            }
            transformed_dfs.append(pd.DataFrame(temp_data))
            group_indicators.extend([idx] * 6)

        # 분해된 모든 특성들을 한 번에 결합합니다.
        transformed_data = pd.concat(transformed_dfs, axis=1)
        return transformed_data, group_indicators

    def _visualize_coefficients(self, chosen_features_by_group, cluster_id, center_point):
        """
        선택된 특성들의 계수를 막대그래프로 시각화하고 파일로 저장합니다.
        
        최적화: plt.show()가 그림을 초기화할 수 있으므로, plt.savefig()를 먼저 호출하도록 순서를 변경했습니다.
                 또한 plt.close()를 호출하여 메모리 누수를 방지합니다.
        """
        plot_data = []
        for _, sub_features in chosen_features_by_group.items():
            for feature_name, coef in sub_features:
                scalar_coef = float(coef)
                plot_data.append({'Feature': str(feature_name), 'Coefficient': scalar_coef})

        if not plot_data:
            print("시각화할 선택된 특성이 없습니다.")
            return

        df = pd.DataFrame(plot_data)
        
        # 계수의 절대값 기준으로 정렬하여 영향력이 큰 특성을 확인하기 쉽게 합니다.
        df = df.sort_values(by='Coefficient', key=abs, ascending=False)
        
        # 시각화할 상위 N개 특성만 선택합니다.
        df_display = df.head(self.args.num_features_to_display)
        df_display = df_display.sort_values(by='Coefficient', key=abs, ascending=True)

        plt.figure(figsize=(12, len(df_display) * 0.4 + 2))
        colors = ['#3498db' if c > 0 else '#e74c3c' for c in df_display['Coefficient']]
        plt.barh(df_display['Feature'], df_display['Coefficient'], color=colors)

        plt.title(f'Cluster {cluster_id}: center point {center_point} (Feature Coefficients)', fontsize=16)
        plt.xlabel('Coefficient', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 그림을 파일로 먼저 저장한 후 화면에 표시합니다.
        plt.savefig(f'figures/cluster_{cluster_id}_coefficients.png', dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close() # Figure 객체를 메모리에서 명시적으로 닫아줍니다.

    def do_lasso_regression(self):
        """
        클러스터별 Group Lasso 또는 일반 Lasso 회귀분석을 수행합니다.
        """
        self.data_for_regression = self._get_data_for_regression()
        
        for cluster_id, feature_indices in self.c_nearest_points.items():
            print(f"\n>>>클러스터 {cluster_id}에 대한 Lasso 회귀 분석 시작...")

            y_index = feature_indices[0]
            x_indices = feature_indices[1: 1 + self.args.num_nearest_points]

            Y = self.data_for_regression.iloc[:, y_index]
            X = self.data_for_regression.iloc[:, x_indices].copy()
            print(f"   - Y (target) for cluster {cluster_id}:\n{Y.head()}")
            print(f"   - X (features) shape for cluster {cluster_id}: {X.shape}") 
            X_transformed, group_indicators = self._tranform_data_for_lasso(X)

            scaler = StandardScaler()
            X_transformed_scaled = scaler.fit_transform(X_transformed)
            center_point = self.feature_dict[feature_indices[0]]

            if self.args.do_grouplasso:
                print(f"--- 클러스터 {cluster_id} Group Lasso 회귀 분석 시작 ---")
                print('center point:', center_point)
                
                group_lasso = GroupLasso(
                    groups=group_indicators, 
                    group_reg=self.args.group_alpha, 
                    l1_reg=self.args.alpha, 
                    supress_warning=True
                )
                group_lasso.fit(X_transformed_scaled, Y)
                y_pred = group_lasso.predict(X_transformed_scaled)
                group_lasso_coef = group_lasso.coef_

                # Debugging: Print shape and denominator for adj_r2
                n_samples, n_features = X_transformed_scaled.shape
                n_features = n_features/  6
                print(f"   - Debug: X_transformed_scaled shape: {n_samples} samples, {n_features} features")
                print(f"   - Debug: Denominator for Adj R^2: {n_samples} - {n_features} - 1 = {n_samples - n_features - 1}")

                # R^2, adjR^2 계산, (Y, Y_hat) 그림 추가
                r2 = r2_score(Y, y_pred)
                if (n_samples - n_features - 1) > 0:
                    adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
                else:
                    adj_r2 = "N/A" # Adjusted R^2 cannot be calculated
                
                print(f"   - R^2: {r2:.4f}")
                print(f"   - Adjusted R^2: {adj_r2:.4f}")

                plt.figure(figsize=(8, 6))
                sns.regplot(x=Y, y=y_pred, ci=None, line_kws={'color': 'red', 'linestyle': '--'})
                plt.xlabel("Actual Y")
                plt.ylabel("Predicted Y")
                adj_r2_display = f'{adj_r2:.3f}' if not isinstance(adj_r2, str) else str(adj_r2)
                plt.title(f'Cluster {cluster_id} ({center_point}): Actual vs. Predicted (Adj R2: {adj_r2_display})')
                plt.grid(True)
                plt.savefig(f'figures/cluster_{cluster_id}_actual_vs_predicted.png', dpi=300)
                #plt.show()
                plt.close()

                chosen_features_by_group = {}
                nonzero_counter = 0
                
                original_feature_names = X.columns
                transformed_feature_names = X_transformed.columns
                
                # 계수가 0이 아닌 특성들을 분석합니다.
                for i, coef in enumerate(group_lasso_coef):
                    if coef != 0:
                        nonzero_counter += 1
                        group_idx = group_indicators[i]
                        original_name = original_feature_names[group_idx]
                        transformed_name = transformed_feature_names[i]

                        if original_name not in chosen_features_by_group:
                            chosen_features_by_group[original_name] = []
                        chosen_features_by_group[original_name].append((transformed_name, coef))

                if not chosen_features_by_group:
                    print("모든 그룹의 계수가 0이 되어 선택된 특성이 없습니다.")
                    print(f"   (Lasso alpha 값을 줄여보세요: group_alpha={self.args.group_alpha}, l1_reg={self.args.alpha})")
                else:
                    print(f"총 {len(X.columns)}개 그룹 중 {len(chosen_features_by_group)}개 그룹이 선택되었습니다.")
                    self._visualize_coefficients(chosen_features_by_group, cluster_id, center_point)
                
                print(f"   - 살아남은 특성 개수:  {nonzero_counter} (lambda: {self.args.group_alpha}, alpha: {self.args.alpha})")
                print(f"   - 계수가 0인 특성 개수: {len(group_lasso_coef) - nonzero_counter} (lambda: {self.args.group_alpha}, alpha: {self.args.alpha})")

            else:
                # 일반 Lasso 회귀분석 부분
                print(f"--- 클러스터 {cluster_id} Lasso 회귀 분석 시작 ---")
                lasso = Lasso(alpha=self.args.alpha, max_iter=1000)
                lasso.fit(X_transformed_scaled, Y)
                lasso_coef = lasso.coef_
                print(f"Lasso Coefficients: {lasso_coef}")

    def run_regression_analysis(self):
        """
        Statsmodels를 사용하여 클러스터별 OLS 회귀분석을 수행하고 결과를 출력합니다.
        """
        if self.data_for_regression is None:
            self.data_for_regression = self._get_data_for_regression()

        self.regression_models = {}
        self.regression_results = {}

        for cluster_id, feature_indices in self.c_nearest_points.items():
            print(f"\n>>> 클러스터 {cluster_id}에 대한 회귀 분석 시작...")

            y_index = feature_indices[0]
            x_indices = feature_indices[1: 1 + self.args.num_nearest_points]

            Y = self.data_for_regression.iloc[:, y_index].astype(float)
            X = self.data_for_regression.iloc[:, x_indices].copy()
            X['status'] = self.data_for_regression['status']

            # 범주형 변수 'status'를 더미 변수로 변환합니다.
            X = pd.get_dummies(X, columns=['status'], drop_first=True, dtype=float)

            # statsmodels는 절편을 자동으로 추가하지 않으므로, 상수항을 수동으로 추가합니다.
            X_with_const = sm.add_constant(X, has_constant='add')
            
            # feature_dict를 사용하여 숫자 인덱스 컬럼명을 실제 특성 이름으로 변경합니다.
            # get(col, col)을 사용하여 딕셔너리에 없는 컬럼명은 그대로 유지합니다(예: 'const', 'status_normal' 등).
            renamed_columns = {col: self.feature_dict.get(col, col) for col in X_with_const.columns}
            X_with_const.rename(columns=renamed_columns, inplace=True)

            model = sm.OLS(Y, X_with_const)
            results = model.fit()

            self.regression_models[cluster_id] = results
            self.regression_results[cluster_id] = {
                'r2_score': results.rsquared,
                'coefficients': results.params,
                'pvalues': results.pvalues,
                'intercept': results.params['const'],
                'predictions': results.predict(X_with_const)
            }

            print(f"--- 클러스터 {cluster_id} 분석 결과 요약 ---")
            print(results.summary())
            print("-" * 80)

        print("\n모든 클러스터에 대한 회귀 분석이 완료되었습니다.")