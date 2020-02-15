"""
Augmentation functions designed for object detection/tracking tasks
"""

import tensorflow as _tf
import numpy as _np
import copy as _copy
import warnings as _warnings
import tensorflow_datasets as _tfds

from . import plot as _plot

def augment(split, 
             ds_slice,
             dtype = _tf.float32,
             flip_left_right = True,
             flip_up_down = True,
             rot90 = True,
             smallest_side_length = None,
             fetch_RPN_anchors = False,
             verbose = 0):
    """
    Data/Image augmentation function for a single slice of data (image + bounding box)
    
    Arguments:
    ----------
        split: string. 'train' or 'test'
            - if split='test', only dtype will be applied
        ds_slice: slice of data from the dataset generator
        dtype: dtype to which the image will be converted to
        flip_left_right: boolean. Whether or not to randomly flip left/right
        flip_up_down: boolean. Whether or not to randomly flip up/down
        rot90: boolean. Whether or not to randomly rotate the image 90 degress
        smallest_side_length: int or None. number of pixels the smallest side of the image should be resized to. If None, then no resizing will be performed
        verbose: print-out verbosity. 
            - if >=2, the augmented image with bounding boxes will be plotted
        fetch_RPN_anchors:
    Returns:
    --------
        ds_slice: the slice of data with augmentation applied
    """
    _warnings.filterwarnings('ignore')
    
    #ds_slice = _copy.deepcopy(ds_slice)
    
    #ds_slice = _tfds.as_numpy(ds_slice)
    
    valid_splits = ['train','test']
    assert(split in valid_splits), split+' is not a valid split. Valid splits include: '+ str(split)
    
    assert('dict' in str(type(ds_slice))), 'The dataset generator must return a dictionary for each slice. Received type(ds_slice): '+str(type(ds_slice))
    
    required_keys = ['image','objects']
    for key in required_keys:
        assert(key in list(ds_slice.keys())), 'The dataset slice dictionary is missing '+key+'. The required keys are:'+str(required_keys)
    
    #Fetch image
    img = ds_slice['image']
    
    shape = img.shape
    assert(len(shape)==3), 'The ds_slice["image"] must be of shape (height, width, color). Received img.shape:'+str(shape)
    
    #Fetch Objects
    objects = ds_slice['objects']
    assert('dict' in str(type(objects))), 'The ds_slice["objects"] must be of type "dict". Received: '+str(type(objects))
    
    required_keys = ['bbox','label']
    for key in required_keys:
        assert(key in list(objects.keys())), 'The ds_slice["objects"] dictionary is missing '+key+'. The required keys are:'+str(required_keys)
        
    bboxes = objects['bbox']
        
#     max_ = _tf.math.reduce_max(objects['bbox'])
    
#     assert(max_<=1.0), 'The bbox coordinates should be floats indicating the relative position of the bounding box (i.e. 0..1). Received np.max(objects["bbox"]):'+str(max_)
    
    #Fetch Random Choices for each augmentation
    choices={'flip_left_right':_np.random.choice([True,False],1)[0],
             'flip_up_down':_np.random.choice([True,False],1)[0],
             'rot90':_np.random.choice([0,1,2,3],1)[0],
            }
    
    def resize(img):
        """
        Fetch the shape dimensions that will yield the specified smallest side length, then resize the passed image

        Arguments:
        ---------
            img_shape: (height, width, color) dimensions of the image which will be resized
            smallest_side_length: int. number of pixels the smallest side of the image should be resized to

        Returns
        -------
            img_resize_shape: (height, width, color) dimensions to which the image should be resized to satisfy the smallest_size_length argument passed.
        """
        
        if type(smallest_side_length)!=type(None):

            height = img.shape[0]
            width = img.shape[1]

            if width <= height:
                f = float(smallest_side_length) / width
                resized_height = int(f * height)
                resized_width = smallest_side_length
            else:
                f = float(smallest_side_length) / height
                resized_width = int(f * width)
                resized_height = smallest_side_length


            img_resize_shape = (resized_height, resized_width, img.shape[2])
        
            img = _tf.image.resize(img, size = img_resize_shape[:2], method = 'bicubic')
            
        max_ = _tf.math.reduce_max(img)
        if max_>=1.:
            img = img/255.

        return img
    
    #Augment the image
    if split == 'train':
        if flip_left_right and choices['flip_left_right']:
            if verbose>=1: print('flip_left_right applied')
            img = _tf.image.flip_left_right(img)

        if flip_up_down and choices['flip_up_down']:
            if verbose>=1: print('flip_up_down applied')
            img = _tf.image.flip_up_down(img)

        if rot90:
            k = choices['rot90']
            if verbose>=1: print('rot90 applied with k=',k,'(i.e.', k*90,'degree rotation)')
            img = _tf.image.rot90(img, k = k)

    if type(smallest_side_length)!=type(None):
        if verbose>=1: print('resizing to smallest_side_length:', smallest_side_length)

    img = _tf.py_function(resize,
                           inp = [img],
                           Tout = dtype)

        
    img = _tf.image.convert_image_dtype(img, dtype)
    
    def bbox_augmentation(bboxes):
        """
        Subprocess of augmentation to run inside tf.py_function (i.e. in eager execution)
        """
        bboxes = _np.array(bboxes)
        if split == 'train':

            if flip_left_right and choices['flip_left_right']:
                for obj_idx in range(bboxes.shape[0]):
                    x1 = bboxes[obj_idx][1]
                    x2 = bboxes[obj_idx][3]
                    bboxes[obj_idx][1] = 1 - x2
                    bboxes[obj_idx][3] = 1 - x1 

            if flip_up_down and choices['flip_up_down']:
                for obj_idx in range(bboxes.shape[0]):
                    y1 = bboxes[obj_idx][0]
                    y2 = bboxes[obj_idx][2]
                    bboxes[obj_idx][0] = 1 - y2
                    bboxes[obj_idx][2] = 1 - y1

            if rot90:
                k = choices['rot90']
                for obj_idx in range(bboxes.shape[0]):
                    for i in range(k):
                        x1 = bboxes[obj_idx][1]
                        x2 = bboxes[obj_idx][3]
                        y1 = bboxes[obj_idx][0]
                        y2 = bboxes[obj_idx][2]

                        bboxes[obj_idx][1] = y2 
                        bboxes[obj_idx][3] = y1

                        bboxes[obj_idx][0] = 1-x1
                        bboxes[obj_idx][2] = 1-x2

        return bboxes
        
    # Augment the bboxes
    bboxes = _tf.py_function(bbox_augmentation, 
                                  inp = [bboxes], 
                                  Tout=dtype)
    
    ds_slice['image'] = img
    ds_slice['objects']['bbox'] = bboxes
    _warnings.filterwarnings('default')
        
    return ds_slice