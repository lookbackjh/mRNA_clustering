import pandas as pd
import numpy as np

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.data_path = args.data_path

    def replace_zeros(self, small_value=1e-6):
        self.data = self.data.replace(0, small_value)

    def get_processed_data(self):
        # Load data from Excel file
        self.data = pd.read_excel(self.data_path, sheet_name='data2', header=0)
        # Select the last 30 columns
        self.feature_dict=self.data['Mature_ID'].to_dict() 
        self.data = self.data.iloc[:, -29:]

         # Store the feature names in a dictionary
        # Replace zeros with a small value
        self.replace_zeros(self.args.small_value)

        


        self.data_normalization()
        # Apply transformations if needed

        if self.args.normalization_method == 'naive':
            pass
        elif self.args.normalization_method == 'log':
            # Apply log transformation
            self.data = self.simple_log_transform()
        elif self.args.normalization_method == 'clr':
            # Apply CLR transformation
            self.data = self.clr_transform()


        return self.data
    
    def clr_transform(self):
        # Apply CLR transformation
        self.data = np.log(self.data / self.data.mean())
        return self.data
    
    def simple_log_transform(self):
        # Apply simple log transformation
        self.data = -np.log(self.data)
        return self.data
    
    def data_normalization(self):
        # Normalize the data
        self.data = self.data.div(self.data.sum(axis=0), axis=1)
        return self.data
