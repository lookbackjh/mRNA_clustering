import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from types import SimpleNamespace

class FDRFeatureSelector:
    """
    A class to select features based on a given threshold.
    (Keeping the original group slicing method)
    """

    def __init__(self, args, data, feature_dict=None):
        self.args = args
        self.data = data
        self.feature_dict = feature_dict if feature_dict is not None else {}

    def fdr_correction(self, pvals: list) -> np.ndarray:
        """
        Apply FDR correction to a list of p-values.
        """
        if pvals is None or len(pvals) == 0:
            return np.array([])
        _, corrected_pvals = fdrcorrection(pvals, alpha=self.args.fdr_threshold)
        return corrected_pvals

    def select_features(self):
        """
        Performs t-tests and FDR correction across three group comparisons
        using vectorized operations for high performance.
        """
        data_normal = self.data.iloc[:, :10]
        data_abnormal = self.data.iloc[:, 10:19]
        data_alzheimer = self.data.iloc[:, 19:29]

        # --- Vectorized t-test (for-loop in do_ttest is removed) ---
        _, comp1_pvals = ttest_ind(data_normal, data_abnormal, axis=1, equal_var=False)
        _, comp2_pvals = ttest_ind(data_normal, data_alzheimer, axis=1, equal_var=False)
        _, comp3_pvals = ttest_ind(data_abnormal, data_alzheimer, axis=1, equal_var=False)

        # --- Apply FDR correction ---
        self.comp1_corrected = self.fdr_correction(comp1_pvals)
        self.comp2_corrected = self.fdr_correction(comp2_pvals)
        self.comp3_corrected = self.fdr_correction(comp3_pvals)

        #print how many features passed the FDR threshold 0.1
        num_features_comp1 = np.sum(self.comp1_corrected < self.args.fdr_threshold)
        num_features_comp2 = np.sum(self.comp2_corrected < self.args.fdr_threshold)
        num_features_comp3 = np.sum(self.comp3_corrected < self.args.fdr_threshold)
        print(f"Number of features passing FDR threshold {self.args.fdr_threshold}:")
        print(f"Comparison 1 (Normal vs. Abnormal): {num_features_comp1}")
        print(f"Comparison 2 (Normal vs. Alzheimer): {num_features_comp2}")
        print(f"Comparison 3 (Abnormal vs. Alzheimer): {num_features_comp3}")

        # i also want you to plot the p-values for each comparison and draw a horizontal line at the FDR threshold
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.scatter(range(len(self.comp1_corrected)), self.comp1_corrected, label='Comparison 1', color='blue')
        plt.axhline(y=self.args.fdr_threshold, color='red', linestyle='--')
        plt.title('Comparison 1: Normal vs. Abnormal')
        plt.xlabel('Feature Index')
        plt.ylabel('FDR-corrected p-value')

        plt.subplot(1, 3, 2)
        plt.scatter(range(len(self.comp3_corrected)), self.comp3_corrected, label='Comparison 3', color='orange')
        plt.axhline(y=self.args.fdr_threshold, color='red', linestyle='--')
        plt.title('Comparison 2: Abnormal vs. Alzheimer')
        plt.xlabel('Feature Index')
        plt.ylabel('FDR-corrected p-value')
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        plt.scatter(range(len(self.comp2_corrected)), self.comp2_corrected, label='Comparison 2', color='green')
        plt.axhline(y=self.args.fdr_threshold, color='red', linestyle='--')
        plt.title('Comparison 3: Normal vs. Alzheimer')
        plt.xlabel('Feature Index')     
        plt.ylabel('FDR-corrected p-value')

        plt.show()
        # Save the results to a file
        plt.savefig('fdr_correction_results.png')




        return self

    def display_sorted_features(self):
        """
        Sorts features by their FDR-corrected p-values for each comparison
        and prints them in 'feature_name: p-value' format.
        (This method remains the same as before)
        """
        if not hasattr(self, 'comp1_corrected'):
            print("Please run the 'select_features' method first to calculate p-values.")
            return

        feature_names = self.data.index
        
        def _print_sorted_results(title, corrected_pvals):
            print(f"\n--- {title} (Sorted by p-value) ---")
            
            feature_pvals = list(zip(feature_names, corrected_pvals))
            sorted_features = sorted(feature_pvals, key=lambda item: item[1])
            
            if not sorted_features:
                print("No features to display.")
                return

            for feature, p_val in sorted_features[:self.args.num_features_to_display]:
                print(f"{self.feature_dict[feature]}: {p_val:.6f}")

        _print_sorted_results("Comparison 1: Normal vs. Abnormal", self.comp1_corrected)
        _print_sorted_results("Comparison 2: Normal vs. Alzheimer", self.comp2_corrected)
        _print_sorted_results("Comparison 3: Abnormal vs. Alzheimer", self.comp3_corrected)

        # Save the selected features to a file