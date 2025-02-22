[![Build Status](https://travis-ci.com/jlnerd/pyDSlib.svg?branch=master)](https://travis-ci.com/jlnerd/pyDSlib)
[![codecov](https://codecov.io/gh/jlnerd/pyDSlib/branch/master/graph/badge.svg)](https://codecov.io/gh/jlnerd/pyDSlib)


# Python Data Science Library (pyDSlib)
__Author: [John T. Leonard](https://www.linkedin.com/in/johntleonard/)__<br>
__Repo: [pyDSlib](https://github.com/jlnerd/pyDSlib)__

Custom modules/classes/methods for various data science, computer vision, and machine learning operations in python
    
## Installing & Importing
In your command line interface (CLI):
```
$ pip install --upgrade pyDSlib
```
After this, the package can be imported into jupyter notebook or python in general via the comman:
```import pyDSlib```


# Modules:
```
pyDSlib.ML
pyDSlib.plot
pyDSlib.img
pyDSlib.video
pyDSlib.file_utils
pyDSlib.summary_tables
pyDSlib.kaggle
```

## Modules Overview

Below, we highlight several of the most interesting modules in more detail.

### pyDSlib.ML
Machine learning module for python focusing on streamlining and wrapping sklearn, xgboost, dask_ml, & tensorflow/keras functions

__pyDSlib.ML Sub-Modules:__
```
pyDSlib.ML.preprocessing 
pyDSlib.ML.model_selection
pyDSlib.ML.NeuralNet
pyDSlib.ML.inspection
pyDSlib.ML.postprocessing
````

The sub-modules within pyDSlib.ML are summarized below:

#### pyDSlib.ML.preprocessing 
Functions related to preprocessing/feature engineering for machine learning

The main class of interest is the ```pyDSlib.ML.preprocessing.Preprocessing_pipe``` class, which iterates through a standard preprocessing sequence and saves the resulting engineered data. The standard sequence is:

1. LabelEncode.categorical_features
2. Scale.continuous_features
    * for Scaler_ID in Scalers_dict.keys()
3. Impute.categorical_features
    * for Imputer_cat_ID in Imputer_categorical_dict[Imputer_cat_ID].keys():<br>
        *for Imputer_iter_class_ID in Imputer_categorical_dict[Imputer_cat_ID].keys():
4. Imputer.continuous_features
    * for Imputer_cont_ID in Imputer_continuous_dict.keys():
        * for Imputer_iter_reg_ID in Imputer_continuous_dict[Imputer_cont_ID].keys():
5. OneHotEncode
6. CorrCoeffThreshold
Finished!
        
#### pyDSlib.ML.model_selection
Functions/classes for running hyperparameter searches across multiple types of models & comparing those models

The main classes of interest are the ```pyDSlib.ML.model_selection.GridSearchCV``` class and the ```pyDSlib.ML.model_selection.BayesianSearchCV``` class, which run hyperparameter GridSearchCV and BayesianSearchCV optimizations across different types of models & compares the results to allow one to find the best-of-best (BoB) model. The ```.fit``` functions for both these classes are compatible with evaluating sklearn models, tensorflow/keras models, and xgboost models. Check out the doc-strings for each class for additional notes on implementation.

#### pyDSlib.ML.NeuralNet
sub-modules/functions/classes for streamlining common neural-net architectures implemented in tensorflow/keras.

The most notetable sub-modules are the ```DenseNet``` and ```Conv2D``` modules, which provide a keras implementation of a general dense neural network & 2D convolutional neural network, where the depth & general architecture of the network s are defined by generic hyperparameters, such that one can easily perform a grid search across multiple neural network architectures.

#### pyDSlib.ML.inspection
Functions to inspect features and/or models after training

#### pyDSlib.ML.postprocessing
ML model outputs postprocessing helper functions


### pyDSlib.plot
This module contains helper functions related to common plotting operations via matplotlib.

The most noteable functions are:

```pyDSlib.plot.corr_matrix()```: Plot a correlation matrix chart

```pyDSlib.plot.ccorr_pareto()```: Plot a pareto bar-chart for 1 label of interest within a correlation dataframe

```pyDSlib.plot.hist_or_bar()```: Iterate through each column in a dataframe and plot the histogram or bar chart for the data.

### pyDSlib.img
This module contains functions/classes related to image analysis, most of which wrap SciKit image functions in some way.

The most noteable functions are: 

```pyDSlib.img.auto_crop.use_edges()```: Use skimage.feature.canny method to find edges in the image passed and autocrop on the outermost edges

```pyDSlib.img.decompose_video_to_img()```: Use cv2 to pull out image frames from a video and save them as png files


### pyDSlib.kaggle
This module contains functions for interacting with kaggle. The simplest and most useful function is:
```
pyDSlib.kaggle.competition_download_files(competition)
```
where ```competition``` is the competition name, such as  "home-credit-default-risk"

### pyDSlib.file_utils
This module contains simple but extremely useful helper functions to save and load standard file types including 'hdf', 'csv', 'json', 'dill'. Essentially the ```save``` and ```load``` functions take care of the boiler plate operations related to saving or loading on the file-types specified above.

# Example Notebooks
Basic notebook examples can be found in the (notebooks)[notebooks] folder. Some examples include:
* [example_ML_NeuralNet_Bert_Word2Vec](notebooks/example_ML_NeuralNet_Bert_Word2Vec.ipynb)
* [example_ML_model_selection_BayesianSearchCV](notebooks/example_ML_model_selection_BayesianSearchCV.ipynb)
* [example_Conv2D_AutoEncoder](notebooks/example_Conv2D_AutoEncoder.ipynb)
* [examples_RCNN](notebooks/examples_RCNN)
     * This folder contains various examples related to region-based Conv. Nets., which are typically used for object detections tasks
     * [example-RCNN-mask-pretrained-coco](example-RCNN-mask-pretrained-coco.ipynb

