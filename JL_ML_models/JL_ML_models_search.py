import sklearn, sklearn.model_selection

import JL_NeuralNet as NeuralNet

import dill, sys, os, shutil
import numpy as np

def save_model_dict(path_model_dir, model_dict):
    if os.path.isdir(path_model_dir)==False:
        os.makedirs(path_model_dir)
    
    path_file = os.path.join(path_model_dir,'model_dict.dill')
    f = open(path_file, 'wb') 
    dill.dump(model_dict, f)
    f.close()
    
def load_model_dict(path_model_dir):
    path_file = os.path.join(path_model_dir,'model_dict.dill')                
    f = open(path_file, 'rb') 
    model_dict = dill.load(f)
    f.close()
    return model_dict
def load_NeuralNet_model_dict(path_model_dir, X_train, y_train, epochs):
    #fetch best params
    path_file = os.path.join(path_model_dir,'best_params_.dill')                
    f = open(path_file, 'rb') 
    best_params_ = dill.load(f)
    f.close()

    #rebuild model_dict
    model_dict = NeuralNet.DenseNet.model_dict(**best_params_)
    model_dict['model'].fit(X_train, y_train, epochs, verbose = 0)
    model_dict['best_model'] = model_dict['model']
    model_dict['best_params'] = best_params_ 
    model_dict['best_cv_score'] = np.nan

    return model_dict

    
def GridSearchCV_single_model(model_dict, X_train, y_train, X_test, y_test, cv, scoring, 
                              path_model_dir, **kwargs):
    """
    Run Grid Search CV on a single model specified by the "key" argument
    """
    if 'compiler' not in model_dict.keys(): #i.e. if you're not running a neural network
        model_dict['GridSearchCV'] = sklearn.model_selection.GridSearchCV(model_dict['model'],
                                                                          model_dict['param_grid'],
                                                                          n_jobs=-1,
                                                                          cv = cv,
                                                                          scoring=scoring,
                                                                          verbose = 1)
        model_dict['GridSearchCV'].fit(X_train,y_train)

    else: #run gridsearch using neural net function
        if scoring == None:
            scoring={'metric': 'loss', 'maximize': False}

        #check kwargs for epochs
        epochs = 100
        for item in kwargs.items():
            if 'epochs' in item[0]: epochs = item[1]

        model_dict['GridSearchCV'] = NeuralNet.search.GridSearchCV(model_dict['compiler'],
                                                                   model_dict['param_grid'],
                                                                   cv = cv,
                                                                   scoring=scoring,
                                                                   epochs =  epochs,
                                                                   path_report_folder = path_model_dir)

        model_dict['GridSearchCV'].fit(X_train,y_train, X_test, y_test)


    model_dict['best_model'] = model_dict['GridSearchCV'].best_estimator_
    model_dict['best_params'] = model_dict['GridSearchCV'].best_params_
    model_dict['best_cv_score'] = model_dict['GridSearchCV'].best_score_
    
    if 'compiler' not in model_dict.keys(): #neural networks save in their own gridsearch function
        save_model_dict(path_model_dir, model_dict)
    
    return model_dict
                      

def GridSearchCV(models_dict, 
                 X_train,
                 y_train, 
                 X_test, 
                 y_test, 
                 cv = 5,
                 scoring=None,
                 metrics = {None:None},
                 retrain = True,
                 path_root_dir = '.',
                 **kwargs):
    """
    metrics: [[key(str), method(sklearn.metrics...)]'
    """
    
    for key in models_dict.keys():
        print('\n----',key,'----')
        
        #define model directory
        path_model_dir = os.path.join(path_root_dir, key)
        print('path_model_dir:',path_model_dir)
        
        if 'Net' not in key:
            path_file = os.path.join(path_model_dir,'model_dict.dill')
        else:
            path_file = os.path.join(path_model_dir,'best_params_.dill')
            
        if retrain or os.path.isfile(path_file)==False:
            models_dict[key] = GridSearchCV_single_model(models_dict[key], 
                                                         X_train, y_train, X_test, y_test, 
                                                         cv, scoring, 
                                                         path_model_dir,
                                                         **kwargs)
            
        else: #reload previously trained model
            if 'Net' not in key:
                models_dict[key] = load_model_dict(path_model_dir)
            else:
                #check kwargs for epochs
                epochs = 100
                for item in kwargs.items():
                    if 'epochs' in item[0]: epochs = item[1]
                models_dict[key] = load_NeuralNet_model_dict(path_model_dir, X_train, y_train, epochs)
        
        models_dict[key]['y_test'] = y_test
        models_dict[key]['y_pred'] = models_dict[key]['best_model'].predict(X_test)
        
        if 'Net' not in key:
            models_dict[key]['best_pred_score'] = models_dict[key]['best_model'].score(X_test, y_test)
        else:
            models_dict[key]['best_pred_score'] = models_dict[key]['best_model'].evaluate(X_test, y_test, verbose =0)

        print('\tbest_csv_score:',models_dict[key]['best_cv_score'])
        print('\tbest_pred_score:',models_dict[key]['best_pred_score'])

        for metric_key in metrics.keys():
            if metrics[metric_key] !=None:
                models_dict[key][metric_key] = metrics[metric_key](y_test, models_dict[key]['y_pred'])
                print('\t',metric_key,':',models_dict[key][metric_key])
                
        if 'Net' not in key:
            save_model_dict(path_model_dir, models_dict[key])
        else:
            model_dict_subset = models_dict[key].copy()
            for key in models_dict[key].keys():
                if key not in ['y_test','y_pred','best_pred_score'] +list(metrics.keys()):
                    model_dict_subset.pop(key)
                    
    return models_dict