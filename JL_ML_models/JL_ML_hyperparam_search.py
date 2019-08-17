import sklearn, sklearn.model_selection

import JL_NeuralNet as NeuralNet

def GridSearchCV(models_dict, 
                     X_train,
                     y_train, 
                     X_test, 
                     y_test, 
                     cv = 5,
                     scoring=None,
                     metrics = {None:None},
                 **kwargs):
    """
    metrics: [[key(str), method(sklearn.metrics...)]'
    """
    
    print(kwargs.items())
    
    for key in models_dict.keys():
        print('\n----',key,'----')
        
        if 'Net' not in key:
            models_dict[key]['GridSearchCV'] = sklearn.model_selection.GridSearchCV(models_dict[key]['model'],
                                                                              models_dict[key]['param_grid'],
                                                                              n_jobs=-1,
                                                                              cv = cv,
                                                                              scoring=scoring,
                                                                              verbose = 1)
        else: #run gridsearch using neural net function
            models_dict[key]['GridSearchCV'] = NeuralNet.search.GridSearchCV(models_dict[key]['compiler'],
                                                                             models_dict[key]['param_grid'],
                                                                             cv = cv,
                                                                             scoring=scoring)
                                                                             
            
        models_dict[key]['GridSearchCV'].fit(X_train,y_train)
        models_dict[key]['best_model'] = models_dict[key]['GridSearchCV'].best_estimator_
        models_dict[key]['best_params'] = models_dict[key]['GridSearchCV'].best_params_
        models_dict[key]['best_cv_score'] = models_dict[key]['GridSearchCV'].best_score_
        
        models_dict[key]['y_test'] = y_test
        models_dict[key]['y_pred'] = models_dict[key]['best_model'].predict(X_test)
        models_dict[key]['best_pred_score'] = models_dict[key]['best_model'].score(X_test, y_test)

        print('\tbest_csv_score:',models_dict[key]['best_cv_score'])
        print('\tbest_pred_score:',models_dict[key]['best_pred_score'])

        for metric_key in metrics.keys():
            if metrics[metric_key] !=None:
                models_dict[key][metric_key] = metrics[metric_key](y_test, models_dict[key]['y_pred'])
                print('\t',metric_key,':',models_dict[key][metric_key])

    return models_dict