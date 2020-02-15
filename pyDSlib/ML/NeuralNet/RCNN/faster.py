import os as _os

import numpy as _np
import tensorflow as _tf
_keras = _tf.keras

from .. import blocks as _blocks
from .. import losses as _losses
from .. import ResNet50 as _ResNet50

from ._augment import augment as _augment
from . import _RPN_anchors

def _graph(img_shape = (None, None, 3),
           ROI_shape = (None, 4),
           backboneID = 'ResNet50_pretrained',
           RPN_Conv2D_input_filters = 512,
           RPN_Conv2D_input_kernel_size = (3,3),
           n_anchors = 9,
           n_classes = None
          ):
    """
    """
    
    assert(type(n_classes)!=type(None)), 'n_classes cannot be None. You must specify and int. value corresponding to the number of classes for which the model will make predictions/be fit.'
    
    _keras.backend.clear_session()
    
    # Build the backbone of the graph
    graph={}
    if backboneID == 'ResNet50':
        backbone = _ResNet50._graph(img_shape = img_shape,
                                     weights=None,#'imagenet',
                                     initial_filters = 64,
                                     initial_kernel_size = (7,7),
                                     initial_strides = (2,2),
                                     kernel_size = 3,
                                     kernel_initializer = 'he_normal',
                                     BatchNormMomentum = 0.99,
                                     activation = 'relu',
                                     include_top=False, # this drops the DenseNet
                                     )
    
    elif backboneID == "ResNet50_pretrained":
        backbone = _ResNet50._graph(img_shape = img_shape,
                                    weights = 'imagenet',
                                    include_top = False)
        
    # add backbone to main graph with updated keys
    for key in backbone.keys():
        graph[backboneID + '_' + key] = backbone[key]
    
    #fetch the branch node key for the RPN and classifier branch graphs.
    branch_node_name = list(graph.keys())[-1]
    
    # Add the Region Proposal Network (RPN) Branch
    graph = _blocks.RPN(graph,
                       branch_node_name = branch_node_name,
                       RPN_Conv2D_input_filters = RPN_Conv2D_input_filters,
                       RPN_Conv2D_input_kernel_size = RPN_Conv2D_input_kernel_size,
                       n_anchors = n_anchors)
    
    # Add the ROI Network Branch
    graph = _blocks.ROI(graph, 
                           branch_node_name = branch_node_name,
                           ROI_shape = ROI_shape,
                           ROI_pool_size = 7,
                           n_ROIs = 4,
                           n_classes = n_classes,
                           Dense_units = 4096,
                           Dense_activation = 'relu',
                           Dropout_rate = 0.5,
                           Dense_Dropout_repeats = 2
                         )
           
    return graph

def _models(img_shape = (None, None, 3),
              backboneID = 'ResNet50_pretrained',
              RPN_Conv2D_input_filters = 512,
              RPN_Conv2D_input_kernel_size = (3,3),
              n_anchors = 9,
              n_classes = None,
              RPN_optimizer = _keras.optimizers.Adam(learning_rate = 1e-5),
              ROI_optimizer = _keras.optimizers.Adam(learning_rate = 1e-5)
           ):
    """
    Compile the RCNN.faster keras models (model_all, model_RPN, model_RPN_Classifier) for training & Prediction
    
    Arguments:
    ----------
        img_shape: the shape of the input image to be analyzed. 
        backboneID: string ID for 2D conv. net architecture to be used as the backbone for the model. If '_pretrained' is added as a suffix to the backboneID, the weights for the pre-trained model will be used. See the _graph() function doc-string for additional details.
            -Valid options: 'ResNet50', 'ResNet50_pretrained', 
                            'VGG16', 'VGG16_pretrained',
                            'InceptionNet', 'InceptionNet_pretrained',
                            'MobileNet', 'MobileNet_pretrained'
        RPN_Conv2D_input_filters:
        RPN_Conv2D_input_kernel_size:
        n_anchors: int. Number of RPN anchor points to use
        n_classes: int. Number of classes the model will be fit to or make predictions on.
        
    Returns:
        model_all, model_RPN, model_ROI
        
    Notes:
    ------
        RPN: Region Pooling Network
        ROI: Region of Interest
        
        Inputs:
            model_all: [img, ROI]
            model_RPN: img
            model_ROI: [img, ROI]
            
        Outputs:
            model_all: [RPN_Classifier, RPN_Regressor, ROI_Classifier, ROI_Regressor]
            model_RPN: [ROI_Classifier, ROI_Regressor]
            model_ROI: [ROI_Classifier, ROI_Regressor]
            
    """
    assert(type(n_classes)!=type(None)), 'n_classes cannot be None. You must specify and int. value corresponding to the number of classes for which the model will make predictions/be fit.'
    
    #Fetch the computational graph dictionary
    graph = _graph(img_shape,
                    backboneID = backboneID,
                    RPN_Conv2D_input_filters = RPN_Conv2D_input_filters,
                    RPN_Conv2D_input_kernel_size = RPN_Conv2D_input_kernel_size,
                    n_anchors = n_anchors,
                    n_classes = n_classes)
    
    #Instantiate the backbone model
    last_backbone_layer = [key for key in graph.keys() if backboneID in key][-1]
    model_backbone = _keras.Model(inputs = graph[backboneID+'_img_input'],
                                  outputs = graph[last_backbone_layer],
                                  name = 'model_backbone')
    
    #Instantiate the RPN branch of the model
    model_RPN = _keras.Model(inputs = graph[backboneID+'_img_input'],
                             outputs = [graph['RPN_Classifier'],
                                        graph['RPN_Regressor']],
                             name = 'model_RPN')
    
    #Instaniate the ROI branch of the model
    model_ROI = _keras.Model(inputs = [graph[backboneID+'_img_input'], graph['ROI_Input']],
                                         outputs = [graph['ROI_Classifier'], 
                                                    graph['ROI_Regressor']],
                                         name = 'model_ROI')
    
    #Instantiate the complete model (RPN+ROI) for training
    model_all = _keras.Model(inputs = [graph[backboneID+'_img_input'], 
                                                   graph['ROI_Input']],
                         outputs = [graph['RPN_Classifier'], 
                                    graph['RPN_Regressor'],
                                    graph['ROI_Classifier'],
                                    graph['ROI_Regressor']],
                         name = 'model_all')
    
    # define some default hyper params for the loss functions
    RPN_Classifier_Lambda = 1.0
    RPN_Classifier_epsilon = 1e-4
    RPN_Regressor_Lambda = 1.0
    RPN_Regressor_epsilon = 1e-4
    ROI_Classifier_Lambda = 1.0
    ROI_Regressor_Lambda = 1.0
    ROI_Regressor_epsilon = 1e-4
    
    #Compile the models
    model_RPN.compile(optimizer = RPN_optimizer, 
                      loss = [_losses.RPN_Classifier(n_anchors, 
                                                    Lambda = RPN_Classifier_Lambda,
                                                     epsilon = RPN_Classifier_epsilon),
                             _losses.RPN_Regressor(n_anchors,
                                                   Lambda = RPN_Regressor_Lambda,
                                                   epsilon = RPN_Regressor_epsilon)] )
    model_ROI.compile(optimizer = ROI_optimizer,
                      loss = [_losses.ROI_Classifier(Lambda = ROI_Classifier_Lambda),
                              _losses.ROI_Regressor(n_classes-1, #No regression for background class
                                                    Lambda = ROI_Regressor_Lambda, 
                                                    epsilon = ROI_Regressor_epsilon)],
                      metrics = ['accuracy'])
    
    model_all.compile(optimizer='sgd',
                      loss='mae')
                      
    
    return model_all, model_RPN, model_ROI, model_backbone

def _preprocess_batch(split, 
                      ds_gen, 
                      batch_size,
                      dtype = _tf.float32,
                      flip_left_right = True,
                      flip_up_down = True,
                      rot90 = True,
                      verbose = 0,):
    """
    Preprocessing function to manage 
    
    Arguments:
    ----------
        split: string. 'train' or 'test'
        ds_gen: dataset generator built using tfds.as_numpy()
        dtype: dtype to which the image will be converted to
        flip_left_right: boolean. Whether or not to randomly flip left/right
        flip_up_down: boolean. Whether or not to randomly flip up/down
        rot90: boolean. Whether or not to randomly rotate the image 90 degress
        verbose: print-out verbosity. 
        
        
    """
    
    valid_splits = ['train','test']
    assert(split in valid_splits), split+' is not a valid split. Valid splits include: '+ str(split)
    
    while True:
        for i in range(batch_size):
            
            ds_slice = next(ds_gen)
            
            ds_slice = _augment(split, ds_slice,
                                dtype = dtype,
                                 flip_left_right = flip_left_right,
                                 flip_up_down = flip_up_down,
                                 rot90 = rot90,
                                 verbose = verbose,
                                 labels = labels)
            
            
    
            

class model():
    """
    Instantiate a full RCNN model on which fit and predict operations may be performed
    """
    
    def __init__(self,
                 img_shape = (None, None, 3),
                 backboneID = 'ResNet50_pretrained',
                 RPN_Conv2D_input_filters = 512,
                 RPN_Conv2D_input_kernel_size = (3,3),
                 n_anchors = 9,
                 n_classes = None,
                 RPN_optimizer = _keras.optimizers.Adam(learning_rate = 1e-5),
                 ROI_optimizer = _keras.optimizers.Adam(learning_rate = 1e-5),
                 path_model_dir = None):
        
        """
        Arguments:
        ----------
            img_shape: the shape of the input image to be analyzed. For most cases, just use (None, None, 3) to allow evaluation of an arbitrary image size 
            backboneID: string ID for 2D conv. net architecture to be used as the backbone for the model. If '_pretrained' is added as a suffix to the backboneID, the weights for the pre-trained model will be used. See the _graph() function doc-string for additional details.
                -Valid options: 'ResNet50', 'ResNet50_pretrained', 
                                'VGG16', 'VGG16_pretrained',
                                'InceptionNet', 'InceptionNet_pretrained',
                                'MobileNet', 'MobileNet_pretrained'
            RPN_Conv2D_input_filters:
            RPN_Conv2D_input_kernel_size:
            n_anchors: int. Number of RPN anchor points to use
            n_classes: int. Number of classes the model will be fit to or make predictions on.
            path_model_dir: string, directory path. path to the directory where the model will be saved.
        
        Returns:
        --------
            None. The following objects are added to the model class:
                - _model_all
                - _model_RPN
                - _model_ROI
                - path_model_dir
                - backboneID
        """
        
        self.path_model_dir = path_model_dir
        self.backboneID = backboneID
        self.model_all, self.model_RPN, self.model_ROI, self.model_backbone = _models(
                                                                img_shape = img_shape,
                                                                backboneID = backboneID,
                                                                RPN_Conv2D_input_filters = RPN_Conv2D_input_filters,
                                                                RPN_Conv2D_input_kernel_size = RPN_Conv2D_input_kernel_size,
                                                                n_anchors = n_anchors,
                                                                n_classes = n_classes,
                                                                RPN_optimizer = RPN_optimizer,
                                                                ROI_optimizer = ROI_optimizer)
        
        def fit(self, 
                ds_train_gen,
                ds_test_gen,
                epochs = 50, 
                batch_size = 32,
                n_training_examples = None,
                verbose = 3):
            """
            Fit the RCNN model
            
            Arguments:
            ----------
                ds_train_gen: training dataset generator.
                    calling next(ds_train_gen) should return a dictionary containing the keys: 'image', 
                    
                epochs: int. The number of epochs to run the fit for
                batch_size: int. The batch size per epoch
                n_training_examples: int. The number of training examples in the dataset passed
                verbose: int. print-out verbosity
                
            Notes:
            ------
                See the doc-string of faster._graph() for details on inputs and outputs to each sub-model (model_all, model_RPN, model_ROI)
            """
            assert(type(n_training_examples)!=type(None)), 'n_training examples cannot be None. You must specify an int'
            assert('generator' in str(type(ds_train_gen))), 'ds_train_gen must be a generator. Received type:'+str(type(ds_train_gen))
            assert('generator' in str(type(ds_test_gen))), 'ds_test_gen must be a generator. Received type:'+str(type(ds_test_gen))
            
            RPN_accuracy_RPN_monitor = []
            
            for epoch in range(epochs):
                
                print('Epoch:',epoch,'/',epochs)
                Progbar = _keras.utils.generic_utils.Progbar(n_training_examples)
                
                while True:
                    
                    if len(RPN_accuracy_RPN_monitor) == batch_size and verbose>=3:
                        mean_overlapping_bboxes = float(sum(RPN_accuracy_RPN_monitor))/len(RPN_accuracy_RPN_monitor)
                        RPN_accuracy_RPN_monitor = []
                        
                        if mean_overlapping_bboxes == 0:
                            print('Warning: RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
                            
                    # Generate X (x_img) and label Y ([y_RPN_class, y_RPN_regressor])
                    X, Y, img_meta = next(data_gen)


                                    