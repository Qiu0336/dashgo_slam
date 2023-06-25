

Dashgo_slam
===============

Dashgo_slam is a system used to achieve real-time 2D mapping and localzaition of the EAI DashGo D1 robot platform. It is developed based on ROS and C++, which consists of five ROS packages: dashgo_calibration, dashgo_control, dashgo_localization, dashgo_mapping, and dashgo_navigation, and a lunch file code: launch.py. 

## Citing

If you use this software in an academic work or project, please cite:
```@online{dashgo_slam, author = {Junyin Qiu}, 
   title = {{dashgo_slam} dashgo_slam}, 
  year = 2023, 
  url = {https://github.com/Qiu0336/dashgo_slam.git}, 
  urldate = {2023-06-25} 
 }

## Requirements

Devices:
mobile robot platform: EAI DashGo D1;
Stereo (Inertial) camera sensors: MYNYEYE S1030 or 1040;
Laptop with Linux and ROS system installed.

Softwares:
MYNYEYE SDK, used for capturing the images;
Eigen library, used for Matrix operation;
Pangolin, used for visulization of the trajectory;
Opencv 3.4, used for image processing;
Ceres 2.0 or 2.1, used for optimization;

## How to use



### Classes 

DBoW3 has two main classes: `Vocabulary` and `Database`. These implement the visual vocabulary to convert images into bag-of-words vectors and the database to index images.
See utils/demo_general.cpp for an example

### Load/Store Vocabulary

The file orbvoc.dbow3 is the ORB vocabulary in ORBSLAM2 but in binary format of DBoW3:  https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary
 


