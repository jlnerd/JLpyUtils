"""
Custom neural network loss functions
"""

import tensorflow as _tf
import tensorflow.keras as _keras
import tensorflow.keras.backend as _Kbackend

def RPN_Classifier(n_anchors, Lambda = 1.0, epsilon = 1e-4):
    """
    Loss function for RPN (region pooling network) classification
    
    Arguments:
    ----------
        n_anchors: int. number of anchors
        Lambda: float. weighting hyperparameter
        epsilon: float. weighting hyperparameter
        
    Returns:
    --------
        loss_fxn: The instantiated loss function to which y_true, y_pred can be passed as arguments for evaluation of the loss
        
    Notes:
    ------
        Given n_anchors = 9...
            if y_true[:, :, :, :9] = [0,1,0,0,0,0,0,1,0], then the 2nd and the 8th box is valid which contains pos or neg anchor => isValid
            if y_true[:, :, :, 9:] = [0,1,0,0,0,0,0,0,0], then the 2nd box is pos and 8th box is negative
    """
    
    def loss_fxn(y_true, y_pred):
        loss = (Lambda * _Kbackend.sum(y_true[:, :, :, :n_anchors] 
                                      * _Kbackend.binary_crossentropy( y_pred[:, :, :, :], y_true[:, :, :, n_anchors:])) 
               / _Kbackend.sum(epsilon + y_true[:, :, :, :n_anchors]))
        return loss
    
    return loss_fxn

def RPN_Regressor(n_anchors, Lambda = 1.0, epsilon = 1e-4):
    """
    Loss function for RPN (region pooling network) classificaiton
    
    Arguments:
    ----------
        n_anchors: int. number of anchors
        Lambda: float. weighting hyperparameter
        epsilon: float. weighting hyperparameter
        
     Returns:
    --------
        loss_fxn: The instantiated loss function to which y_true, y_pred can be passed as arguments for evaluation of the loss
    
    Notes:
    ------
        The loss_fxn returned is essentially a smooth L1 loss function of form:
            0.5*x*x (if x_abs < 1)
            x_abx - 0.5 (otherwise)
    """
    def loss_fxn(y_true, y_pred):
        
        residual = y_true[:,:,:,4*n_anchors:] - y_pred
        
        abs_residual = _Kbackend.abs(residual)
        
        # if abs_residual <= 1.0, bool_ = 1
        bool_ = _Kbackend.cast(_Kbackend.less_equal(abs_residual, 1.0), _tf.float32)
        
        loss = (Lambda * _Kbackend.sum(y_true[:, :, :, :4 * n_anchors] 
                                       * (bool_ * (0.5 * residual * residual) + (1 - bool_) * (abs_residual - 0.5)))
                / _Kbackend.sum(epsilon + y_true[:, :, :, :4 * n_anchors]))
        return loss
    
    return loss_fxn

def ROI_Classifier(Lambda = 1.0):
    """
    Modified categorical cross entropy loss function for ROI (region of interest) classification in RCNN models
    
    Arguments:
    ----------
        Lambda: float. weighting hyperparameter
        
    Returns:
    --------
        loss_fxn: The instantiated loss function to which y_true, y_pred can be passed as arguments for evaluation of the loss
        
    Notes:
    ------
        The loss function is basically lambda*mean(categorical_crossentropy(y_true[0,:,:], y_pred[0,:,:])))
    """
    def loss_fxn(y_true, y_pred):
        loss = Lambda * _Kbackend.mean(_keras.losses.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
        return loss
    
    return loss_fxn

def ROI_Regressor(n_classes, Lambda = 1.0, epsilon = 1e-4):
    """
    Modified MAE loss function for ROI (region of interest) regression in RCNN models
    
    Arguments:
    ----------
        Lambda: float. weighting hyperparameter
        
    Returns:
    --------
        loss_fxn: The instantiated loss function to which y_true, y_pred can be passed as arguments for evaluation of the loss
        
    Notes:
    ------
        The loss function is basically lambda*mean(categorical_crossentropy(y_true[0,:,:], y_pred[0,:,:])))
    """
    
    def loss_fxn(y_true, y_pred):
        
        residual = y_true[:, :, 4*n_classes:] - y_pred
        
        abs_residual = _Kbackend.abs(residual)
        
        # if abs_residual <= 1.0, bool_ = 1
        bool_ = _Kbackend.cast(_Kbackend.less_equal(abs_residual, 1.0), _tf.float32)
        
        loss = (Lambda * _Kbackend.sum(y_true[:, :, :4*n_classes] 
                                       * (bool_ * (0.5 * residual * residual) + (1 - bool_) * (abs_residual - 0.5))) 
                / _Kbackend.sum(epsilon + y_true[:, :, :4*n_classes]))
        return loss
    
    return loss_fxn
    
        

