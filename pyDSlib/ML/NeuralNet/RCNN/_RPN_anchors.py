"""
Functions related to calculating the anchor points for RCNN models using an RPN
"""

import numpy as _np

#default values shared throughout RCNN modules of API
_anchor_box_sizes = (64, 128, 256) #assumes smallest image size = 300 pixels
_anchor_box_ratios = [[1, 1], 
                     [1./_np.sqrt(2), 2./_np.sqrt(2)], 
                     [2./_np.sqrt(2), 1./_np.sqrt(2)]]
_RPN_stride = 16

def RPN_anchors(ds_slice,
                RPN_stride = _RPN_stride,
                anchor_box_sizes = _anchor_box_sizes,
                anchor_box_ratios = _anchor_box_ratios,
                model_backbone = None,
                verbose = 0
               ):
    """
    Calculate the 'RPN_Classifier' and 'RPN_Regressor' labels (anchors) for the given image which the RCNN network will use for training.
    
    Arguments:
    ----------
        ds_slice: dict. A single slice of data from the dataset of interest.
        RPN_stride: The stride length for the RPN
        anchor_box_sizes: The size of the anchor boxes.
            - The original faster.RCNN paper used (128, 256, 512).
            - The anchor box size should be scaled according to the image size
        anchor_box_ratios: The ratios of the anchor boxes
        model_backbone: The tf.keras model which takes the img as input and outputs the last layer of the backbone portion of the RCNN model (i.e. ResNet, VGG, etc.).
            - This is used to determine the height and width of the RPN anchor outputs
        
    Notes:
    ------
        - If the feature map has shape 38x50=1900, there will be 1900x9=17100 potential anchor points
        - The RPN Classifier is a boolean indicator representing whether or not the predicted box is valid.
    """
    
    n_anchors = len(anchor_box_sizes)*len(anchor_box_ratios)
    if verbose>=1:
        print('n_anchors:',n_anchors)
    n_anchor_ratios = len(anchor_box_ratios)
        
    #Calculate the output map size based on the network architecture
    assert(type(model_backbone)!=type(None)), 'The function cannot run with model_backbone=None, Pass in the model_backbone for the RCNN model under evaluation (see the doc-string for more details)'
    
    img = ds_slice['image']
    output_height, output_width = model_backbone.predict(_np.array([img])).shape[1:3]
    
    # initialise empty outputs
    y_RPN_overlap = _np.zeros((output_height, output_width, n_anchors))
    
    #Is box valid
    y_RPN_Classier = _np.zeros((output_height, output_width, n_anchors))
    
    #Box coordinates
    y_RPN_Regressor = _np.zeros((output_height, output_width, 4*n_anchors))
    
    #Fetch n_bboxes
    objects = ds_slice['objects']
    n_bboxes = len(objects['label'])
    
    n_anchors_for_bbox = _np.zeros(n_bboxes).astype(int)
    best_anchor_for_bbox = -1*_np.ones((n_bboxes, 4)).astype(int)
    best_iou_for_bbox = _np.zeros(n_bboxes).astype(_np.float32)
    best_x_for_bbox = _np.zeros((n_bboxes, 4)).astype(int)
    best_dx_for_bbox = _np.zeros((n_bboxes, 4)).astype(_np.float32)

    