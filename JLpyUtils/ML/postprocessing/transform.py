
def one_hot_proba_to_class(y_proba, proba_threshold = 0.5 ):
    """
    Transform a one-hot encoded style numpy array of probablities into 0, 1 class IDs
    
    Arguments:
    ----------
        y_proba: numpy array
    """
    for i in range(y_proba.shape[1]):
        y_proba[:,i][y_proba[:,i]>=proba_threshold] = 1
        y_proba[:,i][y_proba[:,i]<proba_threshold] = 0
    return y_proba