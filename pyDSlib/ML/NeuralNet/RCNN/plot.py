"""
Advanced plotting functions relevant to RCNN models
"""
import matplotlib.pyplot as _plt
import numpy as _np
import copy as _copy
import cv2 as _cv2
import warnings as _warnings

def img_and_bboxes(ds_slice, labels = None, 
                         tight_layout_rect = (0,0,2,2), 
                         verbose = 0):
        """
        show an image and the bounding boxes for that dataset (ds) slice passed

        Arguments:
        ----------
            ds_slice: a single slice generated from a tfds.as_numpy(ds) object/generator
            labels: list containing the string form of each label corresponding to the index contained in the ds_slice
        """
        _warnings.filterwarnings('ignore')
        ds_slice = _copy.deepcopy(ds_slice)

        # fetch image
        img_name = ds_slice['image/filename']

        img = ds_slice['image']
        img = _np.array(img)
        
        height, width, _ = img.shape

        fig, ax = _plt.subplots(1,1)

        ax.set_title(img_name)

        if verbose>=1:
            print(ds_slice)

        #add bbox's and annotations to img
        objs = ds_slice['objects']
        label = objs['label']
        for i in range(len(label)):

            #build box
            bbox = objs['bbox'][i]

            #build box
            xymin = (int(bbox[1]*width), 
                     int(bbox[0]*height))
            xymax = (int(bbox[3]*width), 
                     int(bbox[2]*height))

            _cv2.rectangle(img, xymin, xymax, (0,255,0), 2)

            #build annotation
            label_idx = objs['label'][i]
            if type(labels)==type(None):
                label_name = str(label_idx)
            else:
                label_name = labels[label_idx]
            _cv2.putText(img, label_name, 
                        (xymin[0]+5,xymin[1]+20),
                        _cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

        #plot img
        ax.imshow(img)
        ax.grid(which='both',visible=False)

        fig.tight_layout(rect=tight_layout_rect)
        _plt.show()
        _warnings.filterwarnings('default')