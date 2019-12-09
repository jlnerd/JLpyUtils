import pytest
import numpy as np

import pyDSlib

class TestTransform():
    
    def test_one_hot_proba_to_class(self):
        #build y_proba matrix
        y_proba = np.zeros((10,4))
        for i in range(y_proba.shape[0]):
            y_proba[i,np.random.randint(0,y_proba.shape[1],1)] = 0.9
        
        y_pred_one_hot = pyDSlib.ML.postprocessing.transform.one_hot_proba_to_class(y_proba)
        
        assert(y_pred_one_hot.max()==1), 'Expected max value of 1 but received '+str(y_pred_one_hot.max())
        assert(y_pred_one_hot.min()==0), 'Expected min value of 0 but received '+str(y_pred_one_hot.min())