

class continuous_features():
    """
    Scale the "continuous_features" specified in headers_dict and contained in the X.
    Arguments:
        X: pandas dataframe
        continuous_headers: list containing the header for the continuous features of interest
        Scaler: sklearn.preprocessing....: defaults: sklearn.preprocessing.StandardScaler()
            - Object specifing the scaler operation the data will be fit and transformed to.
    Returns:
        X, Scaler
    """
    
    import sklearn, sklearn.preprocessing
    
    def __init__(self, Scaler = sklearn.preprocessing.RobustScaler()):
        
        self.Scaler = Scaler

        
    def fit(self, X, continuous_headers):
        
        X = X.copy()

        self.Scaler.fit(X[continuous_headers])
        self.continuous_headers = continuous_headers
        
    def transform(self, X):
        
        import warnings
        import dask
        
        warnings.filterwarnings('ignore')
    
        
        type_X = type(X)
        if type_X==dask.dataframe.core.DataFrame:
            npartitions = X.npartitions
            X = X.compute()

        X[self.continuous_headers] = self.Scaler.transform(X[self.continuous_headers])
        
        if type_X==dask.dataframe.core.DataFrame:
            X = dask.dataframe.from_pandas(X, npartitions=npartitions)
            
        warnings.filterwarnings('default')

        return X

def default_Scalers_dict():
    """
    fetch dictionary containing typical scalers used for transforming continuous data
    """
    import sklearn.preprocessing
    
    Scalers_dict = {'MinMaxScaler':sklearn.preprocessing.MinMaxScaler(),
                    'StandardScaler':sklearn.preprocessing.StandardScaler(),
                    'RobustScaler':sklearn.preprocessing.RobustScaler()}
    return Scalers_dict