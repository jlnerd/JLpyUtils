# JLpy Utils Package
Custom modules/classes/methods for various data science, computer vision, and machine learning operations in python

## Dependancies
* General libraries distributed with Anaconda (pandas, numpy, sklearn, scipy, matplotlib, etc.)
* image/video analysis:
    * cv2 (pip install opencv-python)
* ML_models sub-package dependancies:
    * tensorflow or tensorflow-gpu
    * dill
    
## Importing
To import the package, your notebook/code must either be sitting in the same directory as the "JLpy_utils_package" folder, or the package must be in a directory contained in your list of system path (run ```sys.path``` in python/jupyter notebook). Most of the example codes published in my repo. assume you have cloned this repo/package to your desktop, and we simply add the desktop location to your system path via the command:
```sys.path.append(os.path.join(os.path.expanduser("~"),'Desktop'))```
After this, the package can be imported:
```import JLpy_utils_package as JLutils```

## Modules Overview
There are several modules in this package:
```
JLutils.summary_tables
JLutils.plot
JLutils.img
JLutils.video
JLutils.ML_models
```

```JLutils.summary_tables``` and ```JLutils.plot``` probably aren't that useful for most people, so we won't go into detail on them here, but feel free to check them out if you're curious.

### JLutils.img
The ```JLutils.img``` module contains a number of functions related to image analysis, most of which wrap SciKit image functions in some way. The most interesting functions/classes are the ```JLutils.img.auto_crop....``` and ```JLutils.img.decompose_video_to_img()```. 

The ```auto_crop``` class allows you to automatically crop an image using countours via the ```use_countours``` method, which essentially wraps the function ```skimage.measure.find_contours``` function. Alternatively, the ```use_edges``` method provides cropping based on the ```skimage.feature.canny``` function. Generally, I find the ```use_edges``` runs faster and gives more intuitive autocropping results.

The ```decompose_video_to_img()``` is fairly self explanatory and basically uses cv2 to pull out and save all the frames from a video.

### JLutils.video
...



