"""
Custom Neural Network blocks (groups of layers) generic across multiple architecures
"""
import tensorflow as _tf
import tensorflow.keras as _keras
import tensorflow.keras.layers as _layers

def RPN(graph, 
        branch_node_name = None,
        RPN_Conv2D_input_filters = 512,
        RPN_Conv2D_input_kernel_size = (3,3),
        n_anchors = 9,
        ):
    """
    Regional Proposal Network (RPN) Layer/Block used in RCNN models (faster, mask, etc.)
    
    Arguments:
    ----------
        graph: 2D convolutional neural network graph dictionary, such as VGG or ResNet without the top (Dense) layers.
        branch_node_name: The key in the graph for the node at which the RPN layers will be branched or built on to.
            - If None, the last layer/key of the graph will be used
        RPN_Conv2D_input_filters: filter size for the RPN_Conv2D_input layer
        RPN_Conv2D_input_kernel_size: kernel size for the RPN_Conv2D_input layer
        n_anchors: int. Number of RPN anchor points to use
        
    Returns:
    --------
        graph: Computational graph dictionary with the following layers added to the backbone graph:
            -'RPN_Conv2D_input'
            -'RPN_Classifier': Classifier indicating whether the region is an object or not
            -'RPN_Regressor': bounding-boxes regressor
            
    """
    
    if branch_node_name==None:
        branch_node_name = list(graph.keys())[-1]
    
    name = 'RPN_Conv2D_input'
    graph[name] = _layers.Conv2D(RPN_Conv2D_input_filters,
                                        RPN_Conv2D_input_kernel_size,
                                        padding= 'same',
                                        activation = 'relu',
                                        kernel_initializer='normal', 
                                        name=name)(graph[branch_node_name])
    
    name = 'RPN_Classifier'
    graph[name] = _layers.Conv2D(filters = n_anchors, 
                                       kernel_size = (1,1), 
                                       activation = 'sigmoid',
                                       kernel_initializer = 'uniform',
                                       name = name)(graph['RPN_Conv2D_input'])
    name = 'RPN_Regressor'
    graph[name] = _layers.Conv2D(filters = n_anchors * 4, 
                                       kernel_size = (1,1), 
                                       activation = 'linear',
                                       kernel_initializer = 'zero',
                                       name = name)(graph['RPN_Conv2D_input'])
    
    return graph



class ROI_Pooling(_layers.Layer):
    """
    ROI Pooling Layer/Block for 2D inputs.
    
    Reference: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, K. He, X. Zhang, S. Ren, J. Sun
    
    Arguments:
    ---------
        pool_size: int. Number of Pooling Regions
        n_ROIs: int. Number of Regions of interest
        
    Returns:
    --------
        output: Tensor of shape (1, n_ROIs, pool_Size, pool_Size, n_channels)
    """
    
    def __init__(self, 
                 pool_size = 7,
                 n_ROIs = 4,
                 **kwargs):
        
        self.pool_size = pool_size
        self.n_ROIs = n_ROIs
        
        super(ROI_Pooling, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.n_channels = input_shape[0][3]
        
    def compute_output_shape(self, input_shape):
        return None, self.n_ROIs, self.pool_size, self.pool_size, self.n_channels
    
    def call(self, x, mask = None):
        
        assert(len(x)==2), 'Invalid x value passed. The len(x) must equal 2'
        
        img = x[0] #shape = rows, cols, channels
        
        ROIs = x[1] #shape = (n_ROIS, 4) w/ ordering (x,y,w,h)
        
        input_shape = img.shape
        
        output = []
        for ROI_idx in range(self.n_ROIs):
            
            x = ROIs[0, ROI_idx, 0]
            y = ROIs[0, ROI_idx, 1]
            w = ROIs[0, ROI_idx, 2]
            h = ROIs[0, ROI_idx, 3]
            
            x = _keras.backend.cast(x, 'int32')
            y = _keras.backend.cast(y, 'int32')
            w = _keras.backend.cast(w, 'int32')
            h = _keras.backend.cast(h, 'int32')
            
            # Resize ROI of img to pooling size
            resized = _tf.image.resize(img[:, y:y+h, x:x+w, :],
                                      (self.pool_size, self.pool_size))
            output.append(resized)
        
        output = _keras.backend.concatenate(output, axis = 0)
        output = _keras.backend.reshape(output,
                                        (1,
                                         self.n_ROIs, 
                                         self.pool_size, 
                                         self.pool_size, 
                                         self.n_channels))
        
        #Permute Dimensions (similar to transpose)
        output = _keras.backend.permute_dimensions(output, (0, 1, 2, 3, 4))
        
        return output
                     

def ROI(graph, 
                    branch_node_name = None,
                    ROI_shape = (None, 4),
                    ROI_pool_size = 7,
                    n_ROIs = 4,
                    n_classes = None,
                    Dense_units = 4096,
                    Dense_activation = 'relu',
                    Dropout_rate = 0.5,
                    Dense_Dropout_repeats = 2
                    
                    ):
    """
    Region of interest (ROI) layer/block used in RCNN models (faster, mask, etc.)
     
    Arguments:
    ----------
        graph: 2D convolutional neural network graph dictionary, such as VGG or ResNet without the top (Dense) layers.
        branch_node_name: The key in the graph for the node at which the RPN layers will be branched or built on to.
        
        Dense_units: Number of TimeDistributed Dense units to apply
        Dropout_rate: Dropout rate to apply to TimeDistributed dropout layer following each TimeDistributed Dense layer
        
        
    Returns:
    -------
        graph with ROI layers added with "ROI_Classifier", "ROI_Regressor" as the final outputs from the ROI block
    
    """
    
    assert(type(n_classes)!=type(None)), 'n_classes cannot be None. You must specify and int. value corresponding to the number of classes for which the model will make predictions/be fit.'
    
    if branch_node_name==None:
        branch_node_name = list(graph.keys())[-1]
        assert('RPN' not in branch_node_name), branch_node_name+' is not a valid key. the branch_node_name should refer to the last layer of the backbone conv. net., not an RPN layer'
    
    name = 'ROI_Input'
    graph[name] = _layers.Input(shape = ROI_shape)
        
    name = 'ROI_Pooling'
    graph[name] = ROI_Pooling(ROI_pool_size, n_ROIs)([graph[branch_node_name],
                                                          graph['ROI_Input']])
    
    name = 'ROI_TimeDistributed_Flatten'
    graph[name] = _layers.TimeDistributed(
                        _layers.Flatten(name = name)
                    )(graph['ROI_Pooling'])
    
    #Add Dense + Dropout block
    for layer_idx in range(Dense_Dropout_repeats):
        name = 'ROI_TimeDistributed_Dense_'+str(layer_idx)
        graph[name] = _layers.TimeDistributed(
                            _layers.Dense(units = Dense_units,
                                                activation = Dense_activation,
                                                name = name)
                          )(graph[list(graph.keys())[-1]])
        name = 'ROI_TimeDistributed_Dropout_'+str(layer_idx)
        graph[name] = _layers.TimeDistributed(
                            _layers.Dropout(rate = Dropout_rate )
                          )(graph[list(graph.keys())[-1]])
    
    #build output branches (classifier and regressor)
    name = 'ROI_Classifier'
    graph[name] = _layers.TimeDistributed(
                            _layers.Dense(units = n_classes,
                                          activation = 'softmax',
                                          kernel_initializer='zero',
                                          name = name)
                          )(graph['ROI_TimeDistributed_Dropout_'+str(layer_idx)])
    
    name = 'ROI_Regressor'
    graph[name] = _layers.TimeDistributed(
                            _layers.Dense(units = (4*(n_classes-1)), #No Regression for background class
                                          activation = 'linear',
                                          kernel_initializer='zero',
                                          name = name)
                          )(graph['ROI_TimeDistributed_Dropout_'+str(layer_idx)])
    
    
    return graph
    
    

