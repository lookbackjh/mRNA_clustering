
class FDRFeatureSelector:
    """
    A class to select features based on a given threshold.
    """

    def __init__(self, args,data):
        self.args = args
        self.data=data # note that data is in 3 groups, normal, abnormal and alzheimer.


    def select_features(self) -> list:
        pass
