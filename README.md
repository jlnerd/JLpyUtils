# JLpyUtils
Custom modules/classes/methods for various data science, computer vision, and machine learning operations in python

## Dependancies
* General libraries distributed with Anaconda (pandas, numpy, sklearn, scipy, matplotlib, etc.)
* image/video analysis:
    * cv2 (pip install opencv-python)
* ML_models sub-package dependancies:
    * tensorflow or tensorflow-gpu
    * dill
    
## Installing & Importing
In CLI:
```
$ pip install -upgrade JLpyUtils
```
After this, the package can be imported into jupyter notebook or python in general via the comman:
```import JLpyUtils```

## Modules Overview
There are several modules in this package:
```
JLpyUtils.summary_tables
JLpyUtils.plot
JLpyUtils.img
JLpyUtils.video
JLpyUtils.ML_models
```

```JLpyUtils.summary_tables``` and ```JLpyUtils.plot``` probably aren't that useful for most people, so we won't go into detail on them here, but feel free to check them out if you're curious.

### JLpyUtils.img
The ```JLpyUtils.img``` module contains a number of functions related to image analysis, most of which wrap SciKit image functions in some way. The most interesting functions/classes are the ```JLpyUtils.img.auto_crop....``` and ```JLpyUtils.img.decompose_video_to_img()```. 

The ```auto_crop``` class allows you to automatically crop an image using countours via the ```use_countours``` method, which essentially wraps the function ```skimage.measure.find_contours``` function. Alternatively, the ```use_edges``` method provides cropping based on the ```skimage.feature.canny``` function. Generally, I find the ```use_edges``` runs faster and gives more intuitive autocropping results.

The ```decompose_video_to_img()``` is fairly self explanatory and basically uses cv2 to pull out and save all the frames from a video.

### JLpyUtils.video
...

### JLpyUtils.kaggle
This module contains functions for interacting with kaggle. The simplest function is:
```
JLpyUtils.kaggle.competition_download_files(competition)
```
where ```competition``` is the competition name, such as  "home-credit-default-risk"



