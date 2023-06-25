

Dashgo_slam
===============

Dashgo_slam is a system used to achieve real-time 2D mapping and localzaition of the EAI DashGo D1 robot platform indoors. It is developed based on ROS and C++, which consists of five ROS packages: dashgo_calibration, dashgo_control, dashgo_localization, dashgo_mapping, and dashgo_navigation, and a lunch file code: launch.py.

## Demo

Device: Dashgo D1 and MYNYEYE stereo camera  sensors:
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/device.png)

 Running interface:
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/device.png)

Mapping stage (estimate 2D trajectory with loop closure):
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/result0.png)
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/result1.png)
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/result2.png)
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/result3.png)
where the green line is the corrected trajectory, while pink line is the origin one.
Red pyramid denotes camera position and orientation.

Localization stage:
w/o relocalization, estimated by wheel odometry:
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/w_o_relo.png)

w/ relocalization, correct the pose and reduce the pose drift:
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/w_relo.png)

loop matching:
![image](https://github.com/Qiu0336/dashgo_slam/blob/main/demo/loop_matching.png)


## License

Dashgo_slam is under [GPLv3 license](https://github.com/Qiu0336/dashgo_slam/blob/main/LICENSE).

If you use this software in an academic work or project, please cite:
```@online{dashgo_slam, author = {Junyin Qiu}, 
   title = {{dashgo_slam} dashgo_slam}, 
  year = 2023, 
  url = {https://github.com/Qiu0336/dashgo_slam.git}, 
  urldate = {2023-06-25} 
 }
```

## Requirements

#Devices:

mobile robot platform: EAI DashGo D1

Stereo (Inertial) camera sensors: MYNYEYE S1030 or 1040

Laptop with Linux and ROS system installed

#Softwares:

MYNYEYE SDK, used for capturing the images

Eigen library, used for Matrix operation

Pangolin, used for visulization of the trajectory

Opencv 3.4, used for image processing

Ceres 2.0 or 2.1, used for optimization

## Build

First, build the DBoW3 lib in dashgo_localization and dashgo_mapping packages:
```
cd DBoW
mkdir build && cd build
cmake ..
make -j4
```

Then, build the ros packages
```
cd workspace
catkin_make
```

## How to use

First, run the ROS core and launch the DashGo robot.
```
roscore
python launch.y
```

Then, if the extrinsic parameters between the Wheel and the Cameras haven't been calibrated, just perform calibration.
```
rosrun dashgo_slam dashgo_control
rosrun dashgo_slam dashgo_calibration
```
Do Control the robot to circle in place, the calibrated parameters (x,y) can be obtained.


Mapping: build the map of the indoor environment, note that the map is just used for relocalization, 
```
rosrun dashgo_slam dashgo_control
rosrun dashgo_slam dashgo_mapping
```
When running dashgo_mapping, the odometry is realized by the Wheels, and images just are saved in the system for loop closing, when detecting loop, a 3DoF loop correction is performed to reduce the trajectory drift. After running, the saved map is composed of frame poses, keypoints, descriptors and the tree of BoW.

Localization: loading the map saved in mapping stage, performing visual relocalization or wheel odometry, publishing the final pose of the robot.
```
rosrun dashgo_slam dashgo_localization
```
Dashgo_localization will try to perform relocalization by finding the loop frame in the map, otherwise it will estimate the pose by wheel odometry only. the final pose is published to /pose topic.

Navigation: to be developed. 
```
rosrun dashgo_slam dashgo_navigation
```
Dashgo_navigation subscribes /pose topic published in dashgo_localization. Since the 2D pose is received, some upstream and complicated tasks can be developed and achieved, such as robot navigation, targeted moving, following fixed trajectories and so on.


## Some Running Examples




