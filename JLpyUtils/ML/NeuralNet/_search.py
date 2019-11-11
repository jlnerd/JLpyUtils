import os as _os
import time as _time
import numpy as _np
import sklearn as _sklearn
import sklearn.model_selection as _sklearn_model_selection
import sklearn.metrics as _sklearn_metrics
import tensorflow as _tf

_callbacks = [_tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience =10)]
    
def Kfold(X, y, n_splits = 5, stratified = False ):
    """
    Build train - validation index generator for K fold splits
    """

    if len(y.shape)>1:
        if y.shape[1]>1: #if y is one hot encoded
            y_flat = _np.zeros((y.shape[0],1))

            encoding_idx = 0
            for c in range(y.shape[1]):
                y_flat[y[:,c]==1] = encoding_idx
                encoding_idx+=1
    else: y_flat = y

    if stratified:
        kf = _sklearn_model_selection.StratifiedKFold(n_splits = n_splits, shuffle=True)
    else:
        kf = _sklearn_model_selection.KFold(n_splits=n_splits, shuffle=True)

    kf_Xy_split_idx_gen = kf.split(X)

    return kf_Xy_split_idx_gen 


def cross_val_score(model,
                    batch_size,
                    epochs,
                    X, y,
                    callbacks = None,
                    scoring = None,
                    cv = 4,
                    verbose=0):
    
    """Evaluate a score by cross-validation on a neural network model
    
    Arguments
    ----------
        model : keras-style neural network model object implementing 'fit'
            The object to use to fit the data.
        batch_size: batch size to be used in the fit
        epochs: number of epochs to run the fit for
        callbacks: Any keras callbacks to implement
        X, y: the dataset to be cross-validated
        scoring : string, callable or None, optional, default: None
            A string (see sklearn model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)`` which should return only
            a single value.
            Similar to :func:`cross_validate`
            but only a single metric is permitted.
            If None, the estimator's default scorer (if available) is used.
        cv : int, to specify the number of folds in a `(Stratified)KFold`,
            - An iterable yielding (train, test) splits as arrays of indices.
        verbose : integer, optional
            The verbosity level.
        
    Returns
    -------
        scores : array of float, shape=(len(list(cv)),)
            Array of scores of the estimator for each run of the cross validation.
    """
    if 'dataframe' in str(type(X)).lower():
        X = _np.array(X)
    if 'dataframe' in str(type(y)).lower():
        y = _np.array(y)
        
    #get cv split generator
    kf_Xy_split_idx_gen  = Kfold(X, y, cv)

    scores = []
    time_cv = _time.time() #total cv training time
    c=0
    for train_idx, val_idx in kf_Xy_split_idx_gen:
        
        time_train = _time.time()
        
        X_train_val, X_val = X[train_idx], X[val_idx]
        y_train_val, y_val = y[train_idx], y[val_idx]

        #fit the model
        history = model.fit(x=X_train_val, 
                            y=y_train_val, 
                            validation_data=(X_val, y_val),
                            batch_size=batch_size, 
                            epochs = epochs, 
                            verbose= verbose, 
                            callbacks = callbacks)
        metrics_dict = {}
        for key in history.history.keys():
            if key!='lr':
                metrics_dict[key] = history.history[key][-1]
          
        if scoring == None:
            scores.append(metrics_dict['val_loss'])
        else:
            #append '_score' to be consistant w/ sklearn metric nomenclature
            if '_score' not in scoring: 
                scoring = scoring+'_score'
            
            assert(scoring in _sklearn_metrics.__all__), scoring+' is not a valid scoring. Choose from the list contained in sklearn.metrics.__all__'
                
            scorer = getattr(_sklearn_metrics, scoring)
            y_pred = model.predict(X_val)
            if 'roc_auc' in scoring:
                scores.append(scorer(y_val, y_pred))
            else:
                if 'crossentropy' in model.loss:
                    # convert the probs to 0, 1 values
                    y_pred = (y_pred < 0.5).astype(int)
                
                scores.append(scorer(y_val, y_pred))
        
        if verbose>=3:
            print('\t\tcv_idx:', c,'score:',scores[-1])
            
        time_train = (_time.time() - time_train)/60

        if verbose>=2:
            print('cv Progress:',round((c+1)/cv*100,3),
                  '(train time (mins):',round(time_train,2),')]',end='\r')
        c+=1

    time_cv = (_time.time() - time_cv)/60
    if verbose>=1:
        print('total_cv time:',time_cv)

    return scores
    
    
class GridSearchCV:
    """
    GridSearchCV on keras-based neural nets
    
    Arguments:
    ---------
        
    
    """
    
    def __init__(self, 
                 model, 
                 param_grid, 
                 callbacks = _callbacks,
                 scoring = {'metric': 'loss', 'maximize':False},
                 batch_size = 32,
                 epochs = 100,
                 cv='warn', 
                 path_report_folder = _os.path.abspath('.'),
                 verbose=1):
        
        import sys, os
        from . import utils
        from ... import file_utils
        
        self.param_grid = param_grid
        self.model = model
        self.cv = cv
        self.verbose = verbose
        self.param_grid = param_grid
        self.callbacks = callbacks
        self.batch_size = batch_size
        self.epochs = epochs
        self.scoring = scoring
        
        #import load and saving functions from JL_file_utils
        
        self.load = file_utils.load
        self.save = file_utils.save
        self.load_model = utils.load_model
        self.save_model = utils.save_model
        
        #check assertions
        assert(type(self.scoring)==dict), 'scoring must be a dictionary with a "metric" and "maximize" key'
        
        #append to cv_Scores list
        if self.scoring['metric'] == None: #use validation loss as the score if history 
            self.scoring['metric'] = 'loss'
            
        if self.scoring['metric'] != 'loss':   
            #update param_grid if scoring metrics not already in param grid
            if scoring['metric'] not in self.param_grid['metrics'][0]:
                self.param_grid['metrics'][0].append(scoring['metric'])
        
        #ensure the report folder has consistant nomenclature at its root
        root_name = 'GridSearchCV'
        if root_name not in path_report_folder:
            path_report_folder = os.path.join(path_report_folder,root_name)
        self.path_report_folder = path_report_folder
        
        #build report folder
        if os.path.isdir(self.path_report_folder)==False:
            os.makedirs(self.path_report_folder)
        
        #save the settings to the report folder
        self.save(self.model, 'model', 'dill', self.path_report_folder)
        
        batch_size_epochs_cv_params = {'batch_size':self.batch_size,
                                      'epochs':self.epochs,
                                      'cv':self.cv}
        
        self.save(batch_size_epochs_cv_params, 'batch_size_epochs_cv_params', 'dill', self.path_report_folder)
        
    
    def ParameterGrid(self, param_grid):
        return list(_sklearn_model_selection.ParameterGrid(param_grid))
    
    def fit(self, X_train, y_train, X_test, y_test):
        
        import warnings
        import sys, os
        
        warnings.filterwarnings('ignore')
        
        X_train = _np.array(X_train)
        y_train = _np.array(y_train)
        X_test = _np.array(X_test)
        y_test = _np.array(y_test)
        
        #build parameter grid list
        self.ParameterGrid_dict = {}
        self.ParameterGrid_dict['params'] = self.ParameterGrid(self.param_grid)
        
        if self.verbose >=1:
            print('running', self.cv, 'fold cross validation on',len(self.ParameterGrid_dict['params']),'candidates, totalling',self.cv*len(self.ParameterGrid_dict['params']),'fits')
            print('Scoring using',self.scoring)
        
        cv_verbosity = self.verbose
        if cv_verbosity==1: cv_verbosity = 0
            
        self.ParameterGrid_dict[self.scoring['metric']] = _np.zeros((len(self.ParameterGrid_dict['params'])))
        
        #save to report folder
        self.save(self.ParameterGrid_dict, 'ParameterGrid_dict', 'dill', self.path_report_folder)
        
        #run grid search
        p=0
        time_cv_list = []
        for params in self.ParameterGrid_dict['params']:
            if self.verbose >=1:
                param_sweep_print = '\tParameter sweep progress: '+\
                      str(round((p+1)/len(self.ParameterGrid_dict['params'])*100,3))+\
                      '(total time (mins):'+\
                      str(round(_np.sum(time_cv_list),2))+' of ~'+\
                      str(round(_np.median(time_cv_list)*len(self.ParameterGrid_dict['params']),2))+')'
            
            #build the model
            model_ = self.model(**params)
            if self.verbose >=4:
                model_.summary()
            
            if self.verbose>=3:
                print('\tn_params:',model_.count_params())
                
            scores = cross_val_score()
            
            Score = _np.mean(scores)
            
            if self.verbose>=2:
                print('\tcv Score:',Score,' (cv time (mins):',round(time_cv,2),')\t\t\t')
            self.ParameterGrid_dict[self.scoring['metric']][p] = Score
            
            #save to report folder
            self.save(self.ParameterGrid_dict, 'ParameterGrid_dict', 'dill', self.path_report_folder)
                                               
            p+=1
        
        #determine best score
        if self.scoring['maximize']:
            self.best_score_ = _np.max(self.ParameterGrid_dict[self.scoring['metric']])
        else:
            self.best_score_ = _np.min(self.ParameterGrid_dict[self.scoring['metric']])
            
        #find idx of best score
        best_idx = _np.where(_np.array(self.ParameterGrid_dict[self.scoring['metric']]) == self.best_score_)[0][0]
        
        #fetch best parameters
        self.best_params_ = self.ParameterGrid_dict['params'][best_idx]
        self.save(self.best_params_, 'best_params_', 'dill', self.path_report_folder)
        
        if self.verbose >=1:
            print('best_score_:',self.best_score_,'         ')
            print('best_params_:')
            display(self.best_params_)
        
        #fetch best model
        if self.verbose >=1:
            print('re-fitting best estimator...')
        self.best_estimator_ = self.model(**self.best_params_)
        self.best_estimator_.fit(x=X_train, y=y_train,
                                   validation_data=(X_test, y_test),
                                   batch_size=self.batch_size, 
                                   epochs = self.epochs, 
                                   verbose= cv_verbosity, 
                                   callbacks = self.callbacks)
        
        try:
            self.best_estimator_.save(os.path.join(self.path_report_folder, 'best_estimator_.h5')) 
        except Exception as e:
            print('Exception at save:',e)
        
        if self.verbose >=1:
            print('...Finished')
            
        warnings.filterwarnings('default')
        
        
            