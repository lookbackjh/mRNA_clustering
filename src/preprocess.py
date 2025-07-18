import pandas as pd
import numpy as np
from scipy.stats import gmean
class Preprocess:
    # ... __init__ ...
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.feature_dict = {}

    def get_processed_data(self):
        data = pd.read_excel(self.data_path, sheet_name='data2', header=0)
        self.feature_dict = data['Mature_ID'].to_dict()
        
        # 원본 데이터를 계속해서 변수에 담아 전달
        processed_data = data.iloc[:, -29:]
        processed_data = self._replace_zeros(processed_data, self.args.small_value)

        if self.args.normalization_method == 'log_naive':
            processed_data = self._log1p_transform(processed_data)
        elif self.args.normalization_method == 'clr':
            processed_data = self._clr_transform(processed_data)
        elif self.args.normalization_method == 'prob':
            processed_data = self._sum_normalize_columns(processed_data)
        elif self.args.normalization_method == 'naive':
            pass  # No transformation needed, just use the original data
        elif self.args.normalization_method == 'log_entropy':
            result = pd.DataFrame(0.0, index=processed_data.index, columns=processed_data.columns)

            # 2. 0보다 큰 값들만 선택하는 마스크(mask)를 만듭니다.
            positive_mask = processed_data > 0

            # 3. 마스크가 True인 위치에만 로그 변환을 적용하여 값을 채워 넣습니다.
            result[positive_mask] = -np.log(processed_data[positive_mask])

            processed_data = result
        else:
            raise ValueError(f"Unsupported normalization method: {self.args.normalization_method}")
        

        # 최종적으로 self.data에 할당하거나 바로 반환
        self.data = processed_data
        return self.data

# internal methods for transformations
    def _replace_zeros(self, df, small_value):
        return df.replace(0, small_value)

    def _clr_transform(self, df):
        geometric_mean = gmean(df, axis=1).reshape(-1, 1)
        return np.log(df.divide(geometric_mean, axis=0))

    def _log1p_transform(self, df):
        return np.log1p(df)

    def _sum_normalize_columns(self, df):
        return df.div(df.sum(axis=0), axis=1)