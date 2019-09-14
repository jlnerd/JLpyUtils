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
$ pip install --upgrade JLpyUtils
```
After this, the package can be imported into jupyter notebook or python in general via the comman:
```import JLpyUtils```

## Modules Overview
There are several main sub-modules in this package:
```
JLpyUtils.plot
JLpyUtils.img
JLpyUtils.video
JLpyUtils.ML
JLpyUtils.summary_tables
```

### JLpyUtils.plot
This sub-module contains helper functions related to common plotting operations via matplotlib.

The most noteable functions are:
```JLpyUtils.plot.corr_matrix()```: Plot a correlation matrix chart
```JLpyUtils.plot.ccorr_pareto()```: Plot a pareto bar-chart for 1 label of interest within a correlation dataframe
```JLpyUtils.plot.hist_or_bar()```: Iterate through each column in a dataframe and plot the histogram or bar chart for the data.

### JLpyUtils.img
This sub-module contains functions/classes related to image analysis, most of which wrap SciKit image functions in some way.

The most noteable functions are: 
```JLpyUtils.img.auto_crop.use_edges()```: Use skimage.feature.canny method to find edges in the image passed and autocrop on the outermost edges
```JLpyUtils.img.decompose_video_to_img()```

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



