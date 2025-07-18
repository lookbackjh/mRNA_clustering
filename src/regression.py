from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from group_lasso import GroupLasso
from sklearn.model_selection import LeaveOneOut
from collections import defaultdict

class Regression:
    UNPENALIZED_GROUP_ID = 100000 
    """
    클러스터링 결과를 기반으로 회귀 분석(OLS, Group Lasso)을 수행하는 클래스.
    """
    def __init__(self, args, clustering, feature_dict=None, reference_variables=None, target_features=None):
        self.args = args
        self.clustered_data = clustering.data
        self.c_nearest_points = clustering.nearest_samples_dict
        self.feature_dict = feature_dict if feature_dict is not None else {}
        self.reference_variables = reference_variables if reference_variables is not None else []
        self.target_features = target_features if target_features is not None else []
        self.data_for_regression = None 

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
        data_normal.loc[:, 'status'] = 'normal'
        data_abnormal.loc[:, 'status'] = 'abnormal'
        data_alzheimer.loc[:, 'status'] = 'alzheimer'

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

        # feature_dict의 역매핑을 생성하여 특성 이름으로 인덱스를 찾을 수 있도록 합니다.
        # feature_dict는 {인덱스: 이름} 형태이므로, {이름: 인덱스} 형태로 변환합니다.
        feature_name_to_idx = {name: idx for idx, name in self.feature_dict.items()}

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
            
            # 현재 특성(col)이 reference_variables에 포함되는지 확인합니다.
            # feature_dict를 사용하여 col(숫자 인덱스)에 해당하는 실제 특성 이름을 찾습니다.
            current_feature_name = self.feature_dict.get(col, str(col)) # col이 숫자가 아닐 경우를 대비해 str(col) 사용
            
            if current_feature_name in self.reference_variables:
                # reference_variables에 해당하는 특성은 그룹 지시자를 UNPENALIZED_GROUP_ID로 설정하여 페널티를 받지 않도록 합니다.
                group_indicators.extend([self.UNPENALIZED_GROUP_ID] * 6)
            else:
                group_indicators.extend([idx] * 6)

        # 분해된 모든 특성들을 한 번에 결합합니다.
        transformed_data = pd.concat(transformed_dfs, axis=1)
        return transformed_data, group_indicators

    def _visualize_coefficients(self, chosen_features_by_group, analysis_id, center_point):
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

        plt.title(f'{analysis_id}: center point {center_point} (Feature Coefficients)', fontsize=16)
        plt.xlabel('Coefficient', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 그림을 파일로 먼저 저장한 후 화면에 표시합니다.
        plt.savefig(f'figures/{analysis_id}_coefficients.png', dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close() # Figure 객체를 메모리에서 명시적으로 닫아줍니다.

    def do_lasso_regression(self):
        """
        클러스터링 결과를 기반으로 Group Lasso 회귀분석을 수행합니다.
        main.py에서 target_features가 제공되었든 아니든, Clustering 객체는
        적절한 클러스터(c_nearest_points)를 계산했다고 가정합니다.
        이 메서드는 그 결과를 일관되게 사용하여 각 클러스터에 대한 회귀분석을 수행합니다.
        """
        self.data_for_regression = self._get_data_for_regression()
        
        # feature_dict의 역매핑을 생성하여 특성 이름으로 인덱스를 찾을 수 있도록 합니다.
        feature_name_to_idx = {name: idx for idx, name in self.feature_dict.items()}

        # c_nearest_points 딕셔너리를 순회하며 각 클러스터에 대해 분석을 수행합니다.
        for cluster_id, feature_indices in self.c_nearest_points.items():
            
            if not feature_indices:
                print(f"경고: 클러스터 {cluster_id}에 할당된 피처가 없습니다. 건너뜁니다.")
                continue

            # 클러스터의 첫 번째 포인트를 타겟 변수(Y)로 사용합니다.
            target_feature_idx = feature_indices[0]
            current_target_name = self.feature_dict.get(target_feature_idx, f"Feature_{target_feature_idx}")

            print(f"\n>>> 클러스터 {cluster_id} (타겟: '{current_target_name}')에 대한 Lasso 회귀 분석 시작...")

            # Y는 현재 타겟 변수입니다.
            Y = self.data_for_regression.iloc[:, target_feature_idx]

            # 설명 변수(X)는 클러스터의 나머지 포인트들과 reference_variables를 포함합니다.
            x_indices = feature_indices[1: 1 + self.args.num_nearest_points]
            
            combined_x_indices = set(x_indices)
            for ref_var_name in self.reference_variables:
                if ref_var_name in feature_name_to_idx:
                    combined_x_indices.add(feature_name_to_idx[ref_var_name])
            
            final_x_indices = sorted(list(combined_x_indices))
            
            # 타겟 변수가 설명 변수 목록에 포함되지 않도록 확인합니다.
            if target_feature_idx in final_x_indices:
                final_x_indices.remove(target_feature_idx)

            if not final_x_indices:
                print(f"경고: 클러스터 {cluster_id}에 대한 설명 변수를 찾을 수 없습니다. 건너뜁니다.")
                continue
                
            print(f"   - Target Y: {current_target_name} (Index: {target_feature_idx})")
            print(f"   - Explanatory X indices: {final_x_indices}")

            X = self.data_for_regression.iloc[:, final_x_indices].copy()
            
            X_transformed, group_indicators = self._tranform_data_for_lasso(X)

            scaler = StandardScaler()
            X_transformed_scaled = scaler.fit_transform(X_transformed)
            
            center_point_display = current_target_name

            if self.args.do_grouplasso:
                print(f"--- Group Lasso 회귀 분석 시작 (타겟: {current_target_name}) ---")
                
                group_reg_dict = {}
                for g in set(group_indicators):
                    if g == self.UNPENALIZED_GROUP_ID:
                        group_reg_dict[g] = 0.0
                    else:
                        group_reg_dict[g] = self.args.group_alpha

                group_lasso = GroupLasso(
                    groups=group_indicators, 
                    group_reg=group_reg_dict, 
                    l1_reg=self.args.alpha, 
                    supress_warning=True
                )
                group_lasso.fit(X_transformed_scaled, Y)
                y_pred = group_lasso.predict(X_transformed_scaled)
                group_lasso_coef = group_lasso.coef_

                n_samples = X_transformed_scaled.shape[0]
                n_features_for_adj_r2 = len(final_x_indices)
                
                r2 = r2_score(Y, y_pred)
                if (n_samples - n_features_for_adj_r2 - 1) > 0:
                    adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features_for_adj_r2 - 1))
                else:
                    adj_r2 = "N/A"
                
                print(f"   - R^2: {r2:.4f}")
                print(f"   - Adjusted R^2: {adj_r2 if isinstance(adj_r2, str) else f'{adj_r2:.4f}'}")

                plt.figure(figsize=(8, 6))
                sns.regplot(x=Y, y=y_pred, ci=None, line_kws={'color': 'red', 'linestyle': '--'})
                plt.xlabel("Actual Y")
                plt.ylabel("Predicted Y")
                adj_r2_display = f'{adj_r2:.3f}' if not isinstance(adj_r2, str) else str(adj_r2)
                plt.title(f'Cluster {cluster_id} - Target: {current_target_name} (Adj R2: {adj_r2_display})')
                plt.grid(True)
                plt.savefig(f'figures/cluster_{cluster_id}_target_{current_target_name}_actual_vs_predicted.png', dpi=300)
                plt.close()

                chosen_features_by_group = {}
                nonzero_counter = 0
                
                original_feature_names = X.columns
                transformed_feature_names = X_transformed.columns
                
                for i, coef in enumerate(group_lasso_coef):
                    if coef != 0:
                        nonzero_counter += 1
                        group_idx = group_indicators[i]
                        original_name = self.feature_dict.get(original_feature_names[group_idx], original_feature_names[group_idx])
                        transformed_name = transformed_feature_names[i]

                        if original_name not in chosen_features_by_group:
                            chosen_features_by_group[original_name] = []
                        chosen_features_by_group[original_name].append((transformed_name, coef))

                if not chosen_features_by_group:
                    print("모든 그룹의 계수가 0이 되어 선택된 특성이 없습니다.")
                    print(f"   (Lasso alpha 값을 줄여보세요: group_alpha={self.args.group_alpha}, l1_reg={self.args.alpha})")
                else:
                    print(f"총 {len(X.columns)}개 그룹 중 {len(chosen_features_by_group)}개 그룹이 선택되었습니다.")
                    analysis_id = f"cluster_{cluster_id}_{current_target_name.replace(':', '_')}"
                    self._visualize_coefficients(chosen_features_by_group, analysis_id, center_point_display)
                
                print(f"   - 살아남은 특성 개수:  {nonzero_counter} (lambda: {self.args.group_alpha}, alpha: {self.args.alpha})")
                print(f"   - 계수가 0인 특성 개수: {len(group_lasso_coef) - nonzero_counter} (lambda: {self.args.group_alpha}, alpha: {self.args.alpha})")

            else:
                # 일반 Lasso 회귀분석 부분
                print(f"--- Lasso 회귀 분석 시작 (타겟: {current_target_name}) ---")
                lasso = Lasso(alpha=self.args.alpha, max_iter=1000)
                lasso.fit(X_transformed_scaled, Y)
                lasso_coef = lasso.coef_
                print(f"Lasso Coefficients: {lasso_coef}")

    def perform_loocv_lasso(self):
        """
        Leave-One-Out Cross-Validation (LOOCV)을 사용하여 Group Lasso 회귀분석을 수행하고,
        변수 선택의 안정성을 평가합니다.
        """
        self.data_for_regression = self._get_data_for_regression()
        feature_name_to_idx = {name: idx for idx, name in self.feature_dict.items()}

        for cluster_id, feature_indices in self.c_nearest_points.items():
            if not feature_indices:
                print(f"경고: 클러스터 {cluster_id}에 할당된 피처가 없습니다. 건너뜁니다.")
                continue

            target_feature_idx = feature_indices[0]
            current_target_name = self.feature_dict.get(target_feature_idx, f"Feature_{target_feature_idx}")

            print(f"\n>>> 클러스터 {cluster_id} (타겟: '{current_target_name}')에 대한 LOOCV Group Lasso 분석 시작...")

            Y_full = self.data_for_regression.iloc[:, target_feature_idx]
            
            x_indices = feature_indices[1: 1 + self.args.num_nearest_points]
            combined_x_indices = set(x_indices)
            for ref_var_name in self.reference_variables:
                if ref_var_name in feature_name_to_idx:
                    combined_x_indices.add(feature_name_to_idx[ref_var_name])
            final_x_indices = sorted(list(combined_x_indices))
            if target_feature_idx in final_x_indices:
                final_x_indices.remove(target_feature_idx)

            if not final_x_indices:
                print(f"경고: 클러스터 {cluster_id}에 대한 설명 변수를 찾을 수 없습니다. 건너뜁니다.")
                continue

            X_full = self.data_for_regression.iloc[:, final_x_indices].copy()
            
            # LOOCV를 위한 데이터 변환은 루프 밖에서 한 번만 수행
            X_transformed_full, group_indicators_full = self._tranform_data_for_lasso(X_full)

            loo = LeaveOneOut()
            selected_feature_counts = defaultdict(int)
            total_iterations = 0

            group_reg_dict = {}
            for g in set(group_indicators_full):
                if g == self.UNPENALIZED_GROUP_ID:
                    group_reg_dict[g] = 0.0
                else:
                    group_reg_dict[g] = self.args.group_alpha

            for train_index, test_index in loo.split(X_transformed_full):
                total_iterations += 1
                X_train, X_test = X_transformed_full.iloc[train_index], X_transformed_full.iloc[test_index]
                Y_train, Y_test = Y_full.iloc[train_index], Y_full.iloc[test_index]
                
                # 스케일러는 훈련 데이터에만 fit하고, 훈련/테스트 데이터 모두에 transform
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Group Lasso 모델 학습
                group_lasso = GroupLasso(
                    groups=group_indicators_full, 
                    group_reg=group_reg_dict, 
                    l1_reg=self.args.alpha, 
                    supress_warning=True
                )
                group_lasso.fit(X_train_scaled, Y_train)
                
                # 선택된 변수 기록
                for i, coef in enumerate(group_lasso.coef_):
                    if coef != 0:
                        # 원래 특성 이름으로 매핑하여 저장
                        original_col_idx = X_full.columns[group_indicators_full[i]] # X_full의 원래 컬럼 인덱스
                        original_feature_name = self.feature_dict.get(original_col_idx, str(original_col_idx))
                        selected_feature_counts[original_feature_name] += 1
            
            print(f"--- 클러스터 {cluster_id} (타겟: '{current_target_name}') LOOCV 결과 ---")
            if total_iterations == 0:
                print("LOOCV 반복이 수행되지 않았습니다. 샘플 수가 부족할 수 있습니다.")
                continue

            print(f"총 LOOCV 반복 횟수: {total_iterations}")
            
            # 선택 빈도에 따라 정렬하여 출력
            sorted_selected_features = sorted(selected_feature_counts.items(), key=lambda item: item[1], reverse=True)
            
            print("\n변수 선택 빈도:")
            for feature, count in sorted_selected_features:
                percentage = (count / total_iterations) * 100
                print(f"  - {feature}: {count}회 선택 ({percentage:.2f}%)")
            
            # 안정적으로 선택된 변수 (예: 80% 이상 선택)
            stable_selected_features = {f: c for f, c in selected_feature_counts.items() if (c / total_iterations) * 100 >= 80}
            if stable_selected_features:
                print("\n안정적으로 선택된 변수 (80% 이상):")
                for feature, count in stable_selected_features.items():
                    print(f"  - {feature} ({count}회)")
            else:
                print("\n80% 이상 안정적으로 선택된 변수가 없습니다.")

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
            X_with_const_np = sm.add_constant(X, has_constant='add')
            
            # NumPy 배열을 DataFrame으로 변환하여 rename 메소드를 사용할 수 있도록 합니다.
            # 컬럼 이름은 기존 X의 컬럼과 'const'를 합쳐서 사용합니다.
            X_with_const = pd.DataFrame(X_with_const_np, columns=X.columns.tolist() + ['const'], index=X.index)
            
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