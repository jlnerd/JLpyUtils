from __init__ import *

import sklearn, sklearn.linear_model, sklearn.tree, sklearn.neighbors

import JL_NeuralNet as NeuralNet

class models_dict():
    """
    fetch dictionaries containing sklearn model objects and relevant hyperparameter grid dictionaries for regression or classification models.
    """
    def regression(n_features, n_labels, NeuralNets=True):
        
        #define models
        models_dict = {}
        models_dict['Linear'] = {'model':sklearn.linear_model.LinearRegression()}
        models_dict['DecisionTree'] = {'model':sklearn.tree.DecisionTreeRegressor()}
        models_dict['RandomForest'] = {'model': sklearn.ensemble.RandomForestRegressor()}
        models_dict['SVM'] = {'model':sklearn.svm.SVR()}
        models_dict['KNN'] = {'model': sklearn.neighbors.KNeighborsRegressor() }
        
        #define parameter grid for grid search
        models_dict['Linear']['param_grid'] =       {'normalize': [False,True]}

        models_dict['DecisionTree']['param_grid'] = {'criterion':      ['mse','friedman_mse','mae'],
                                                     'splitter':       ['best','random'],
                                                     'max_depth':      [None,5,10,100],
                                                     'max_features':   [None,0.25,0.5,0.75,1.],
                                                     'max_leaf_nodes': [None,10,100]}

        models_dict['RandomForest']['param_grid'] = {'n_estimators':[10,100,1000],
                                                     'criterion':models_dict['DecisionTree']['param_grid']['criterion'],
                                                     'max_depth':models_dict['DecisionTree']['param_grid']['max_depth'],
                                                     'max_features':models_dict['DecisionTree']['param_grid']['max_features'],
                                                     'max_leaf_nodes':models_dict['DecisionTree']['param_grid']['max_leaf_nodes']}
        
        models_dict['GradBoost'] = {'model':sklearn.ensemble.GradientBoostingRegressor()}
        models_dict['GradBoost']['param_grid'] = {'loss':['ls', 'lad', 'huber', 'quantile'],
                                                  'learning_rate':[0.01, 0.1, 1],
                                                  'n_estimators':[10, 100, 1000],
                                                  'subsample':[1.0,0.8,0.5],
                                                  'criterion':["friedman_mse",'mse','mae'],
                                                  'max_depth':[None, 5, 10],
                                                  }
        

        models_dict['SVM']['param_grid'] =          {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                                                     'gamma':['auto','scale']}

        models_dict['KNN']['param_grid'] =         {'n_neighbors':[5, 10, 100],
                                                    'weights':['uniform','distance'],
                                                    'algorithm':['ball_tree','kd_tree','brute']}
        if NeuralNets:
            models_dict['DenseNet'] = NeuralNet.DenseNet.model_dict(n_features=n_features,
                                                                     n_labels = n_labels)
        return models_dict
        
        
    def classification():
        #define models_dict
        models_dict = {}
        models_dict['DecisionTree'] = {'model':sklearn.tree.DecisionTreeClassifier()}
        models_dict['RandomForest'] = {'model': sklearn.ensemble.RandomForestClassifier()}
        models_dict['SVM'] = {'model':sklearn.svm.SVC(probability=True)}
        models_dict['KNN'] = {'model': sklearn.neighbors.KNeighborsClassifier() }

        #define param grid for grid search
        models_dict['DecisionTree']['param_grid'] = {'criterion':['gini','entropy'],
                                                  'max_depth':[None,1,10,100],
                                                  'max_features':[None,0.25,0.5,0.75,1.],
                                                  'max_leaf_nodes':[None,10,100]}
        models_dict['RandomForest']['param_grid'] = {'n_estimators':[10,100,1000],
                                              'criterion':models_dict['DecisionTree']['param_grid']['criterion'],
                                              'max_depth':models_dict['DecisionTree']['param_grid']['max_depth'],
                                              'max_features':models_dict['DecisionTree']['param_grid']['max_features'],
                                              'max_leaf_nodes':models_dict['DecisionTree']['param_grid']['max_leaf_nodes']}
        models_dict['SVM']['param_grid'] = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                                        'gamma':['auto','scale']}
        models_dict['KNN']['param_grid'] = {'n_neighbors':[5,10,100],
                                        'weights':['uniform','distance'],
                                        'algorithm':['ball_tree','kd_tree','brute']}

        models['GradBoost'] = {'model':sklearn.ensemble.GradientBoostingClassifier(random_state=seed)}

#         models['GradBoost']['param_grid'] = {'loss':['deviance','exponential'],
#                                             'learning_rate':[0.01, 0.1, 1],
#                                              'n_estimators':[10, 100, 1000],
#                                              'subsample':[1.0,0.8,0.5],
#                                               'max_depth':models['DecisionTree']['param_grid']['max_depth'],
#                                               'max_features':models['DecisionTree']['param_grid']['max_features'],
#                                               'max_leaf_nodes':models['DecisionTree']['param_grid']['max_leaf_nodes']}


        return models_dict