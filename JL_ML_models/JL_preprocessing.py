import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os, warnings, shutil, sklearn, scipy
import sklearn.preprocessing, sklearn.model_selection, sklearn.impute, sklearn.tree, sklearn.ensemble, sklearn.neighbors

class LabelEncoder():
    def categorical_features(df, headers_dict, 
                                          verbose = 0, 
                                          ):
        """
        Arguments:
            df: pandas dataframe
            headers_dict: dictionary containing "categorical_features" key with a list of the headers for each categorical feature in the df. numeric categorical features will not be encoded
            verbose: verbosity index.
        """
        df = df.copy()
        headers_dict = headers_dict.copy()
        
        assert('categorical_features' in list(headers_dict.keys())), 'headers_dict missing "categorical_features" key'

        #fetch the non-numeric categorical headers which will be encoded
        headers_dict['LabelEncodings'] = [header for header in headers_dict['categorical_features'] if pd.api.types.is_numeric_dtype(df[header])==False]
        if verbose>=1: 
            print("headers_dict['LabelEncodings']:\n", headers_dict['LabelEncodings'])

        #build label encoder
        LabelEncoders = {}
        for header in headers_dict['LabelEncodings']:
            LabelEncoders[header] = sklearn.preprocessing.LabelEncoder()

            df[header] = df[header].fillna('missing_value')

            if verbose:
                print(df[header].unique())

            #fetch unique values and ensure 'missing_value' is encoded so that the LabelEncoders can encode test sets
            uniques = list(df[header].sort_values().unique())+['missing_value']
            LabelEncoders[header].fit(uniques)

            #update df
            df[header] = LabelEncoders[header].transform(df[header])

            #fill back in nan values (run imputing as a seperate step)
            warnings.filterwarnings('ignore')
            nan_encoding = LabelEncoders[header].transform(['missing_value'])[0]
            df[header][df[header]==nan_encoding] = np.nan
            warnings.filterwarnings('default')

        return df, headers_dict, LabelEncoders
    
class impute():
    def categorical_features(df, 
                                headers_dict, 
                                strategy = 'most_frequent', 
                                estimator = None,
                                verbose= 0):
        """
        Impute (fill nan) values for categorical features

        Arguments:
            df: pandas dataframe. If strategy = 'iterative', then all categorical features must be label encoded in a previous step, with nan values remaining after encoding.
            strategy : The imputation strategy.
                - If If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data. fill_value will be 0 when imputing numerical data and “missing_value” for strings or object data types.
                - If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data.
                - If 'iterative', then use sklearn.imputer.IterativeImputer with the specified estimator
            estimator: sklearn estimator object
                The estimator to be used if 'iterative' strategy chosen
        Note: sklearn.impute.IterativeImputer has a number of other options which could be varied/tuned, but for simplicity we just use the defaults
        """
        warnings.filterwarnings('ignore')

        df = df.copy()
        headers_dict = headers_dict.copy()

        if strategy in ['most_frequent','constant']:
            Imputer = sklearn.impute.SimpleImputer(strategy=strategy,
                                                   verbose = verbose)

        if strategy == 'iterative':
            n_nearest_features = np.min([10, len(headers_dict['categorical_features'])]) #use less than or equal to 10 features
            Imputer = sklearn.impute.IterativeImputer(estimator= estimator, 
                                                      initial_strategy = 'most_frequent',
                                                      verbose = verbose,
                                                      n_nearest_features = n_nearest_features)
            
        #create a dummy nan row to ensure any dataset containing nan for any of the features can be transformed
        df_nans = pd.DataFrame(np.array([[np.nan for header in headers_dict['categorical_features']]]), 
                               columns =  headers_dict['categorical_features'])
        df_fit = pd.concat((df[headers_dict['categorical_features']],df_nans))
                
        Imputer.fit(df_fit)

        df[headers_dict['categorical_features']] = Imputer.transform(df[headers_dict['categorical_features']])


        #ensure imputation worked correctly
        for header in headers_dict['categorical_features']:
            assert(len(df[df[header].isna()])==0), 'Found nan value for '+ header +' after imputing'

        warnings.filterwarnings('default')
        return df, headers_dict, Imputer

    def continuous_features(df, 
                            headers_dict, 
                            strategy = 'median', 
                            estimator = None,
                            verbose= 0):
        """
        Impute (fill nan) values for continuous features
        
        Arguments:
            df: pandas dataframe. If strategy = 'iterative', then all categorical features must be label encoded in a previous step, with nan values remaining after encoding.
            strategy : The imputation strategy.
                - If If “constant”, then replace missing values with fill_value. fill_value will be 0 when imputing numerical data.
                - If "most_frequent", then replace missing using the most frequent value along each column.
                - If 'iterative', then use sklearn.imputer.IterativeImputer with the specified estimator
            estimator: sklearn estimator object
                The estimator to be used if 'iterative' strategy chosen
            Note: sklearn.impute.IterativeImputer has a number of other options which could be varied/tuned, but for simplicity we just use the defaults
        """
        warnings.filterwarnings('ignore')
        df = df.copy()
        headers_dict = headers_dict.copy()

        if strategy in ['most_frequent', 'constant', 'mean', 'median']:
            Imputer = sklearn.impute.SimpleImputer(strategy=strategy,
                                                   verbose = verbose)
        if strategy == 'iterative':
            n_nearest_features = np.min([10, len(headers_dict['continuous_features'])]) 
            Imputer = sklearn.impute.IterativeImputer(estimator= estimator, 
                                                      initial_strategy = 'most_frequent',
                                                      verbose = verbose,
                                                      n_nearest_features = n_nearest_features)
        #create a dummy nan row to ensure any dataset containing nan for any of the features can be transformed
        df_nans = pd.DataFrame(np.array([[np.nan for header in headers_dict['continuous_features']]]), 
                               columns =  headers_dict['continuous_features'])
        df_fit = pd.concat((df[headers_dict['continuous_features']],df_nans))

        Imputer.fit(df_fit)

        df[headers_dict['continuous_features']] = Imputer.transform(df[headers_dict['continuous_features']])

        #ensure imputation worked correctly
        for header in headers_dict['continuous_features']:
            assert(len(df[df[header].isna()])==0), 'Found nan value for '+ header +' after imputing'

        warnings.filterwarnings('default')
        return df, headers_dict, Imputer

    def fetch_typical_iterative_estimators():
        #focus on BayesianRidge (sklearn default) and RandomForest, since they generally perform better than simple linear or DecisionTree and scale better than KNN
        return [#sklearn.linear_model.LinearRegression(n_jobs=-1),
                sklearn.linear_model.BayesianRidge(),
                #sklearn.neighbors.KNeighborsRegressor(n_jobs=-1)
                #sklearn.tree.DecisionTreeRegressor(),
                sklearn.ensemble.RandomForestRegressor(n_jobs=-1)]

    def validation_test(df, headers_dict, verbose =1 ):
        """
        Iterate over impute_categorical_feature and impute_continuous_features options & ensure everything works for this particular dataset
        """
        
        print('------running impute.continuous_features validation-------')
        for strategy in ['mean','median','iterative']:
            print('strategy:',strategy,)

            if strategy in ['most_frequent','mean','median']:
                df_imputed, headers_dict, Imputer = impute.continuous_features(df, 
                                                                            headers_dict, 
                                                                            strategy = strategy, 
                                                                            estimator = None,
                                                                            verbose = verbose)
            else:
                for estimator in impute.fetch_typical_iterative_estimators():
                    print('estimator:',estimator)

                    df_imputed, headers_dict, Imputer = impute.continuous_features(df, 
                                                                            headers_dict, 
                                                                            strategy = strategy, 
                                                                            estimator = estimator,
                                                                            verbose = verbose)
                    
        print('------running impute.categorical_features validation-------')
        for strategy in ['most_frequent', 'iterative']:
            print('strategy:',strategy,)

            if strategy == 'most_frequent':
                df_imputed, headers_dict, Imputer = impute.categorical_features(df, 
                                                                    headers_dict, 
                                                                    strategy = strategy, 
                                                                    estimator = None,
                                                                    verbose = verbose)
            else:
                for estimator in impute.fetch_typical_iterative_estimators():
                    print('estimator:',estimator)

                    df_imputed, headers_dict, Imputer = impute.categorical_features(df, 
                                                                        headers_dict, 
                                                                        strategy = strategy, 
                                                                        estimator = estimator,
                                                                        verbose = verbose)
                    
                    
      
        
        print('\nall imputation options validated!')