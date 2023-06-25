

Dashgo_slam
=====


Dashgo_slam is an improved version of the DBow2 library, an open source C++ library for indexing and converting images into a bag-of-word representation. It implements a hierarchical tree for approximating nearest neighbours in the image feature space and creating a visual vocabulary. DBoW3 also implements an image database with inverted and direct files to index images and enabling quick queries and feature comparisons. The main differences with the previous DBow2 library are:

  * DBoW3 only requires OpenCV.  DBoW2 dependency of DLIB is been removed.
  * DBoW3 is able to use both binary and floating point descriptors out of the box. No need to reimplement any class for any descriptor.
  * DBoW3 compiles both in linux and windows.  
  * Some pieces of code have been rewritten to optimize speed. The interface of DBoW3 has been simplified.
  * Possibility of using binary files. Binary files are 4-5 times faster to load/save than yml. Also, they can be compressed.
  * Compatible with DBoW2 yml files

## 
## Citing

If you use this software in an academic work or project, please cite:
```@online{DBoW3, author = {Junyin Qiu}, 
   title = {{Dashgo_slam} Dashgo_slam}, 
  year = 2023, 
  url = {https://github.com/Qiu0336/dashgo_slam.git}, 
  urldate = {2023-06-25} 
 } 
```


## Installation notes
 
DBoW3 requires OpenCV, MYNYEYE  only.

For compiling the utils/demo_general.cpp you must compile against OpenCV 3. If you have installed the contrib_modules, use cmake option -DUSE_CONTRIB=ON to enable SURF.

## How to use

Check utils/demo_general.cpp

### Classes 

DBoW3 has two main classes: `Vocabulary` and `Database`. These implement the visual vocabulary to convert images into bag-of-words vectors and the database to index images.
See utils/demo_general.cpp for an example

### Load/Store Vocabulary

The file orbvoc.dbow3 is the ORB vocabulary in ORBSLAM2 but in binary format of DBoW3:  https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary
 


