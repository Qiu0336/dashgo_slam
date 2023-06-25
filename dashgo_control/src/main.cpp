#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
#include <turtlesim/Pose.h>
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "util/keyborad.h"


int main(int argc, char **argv)
{

    KeyboardInput* mpkeyboardinput = new KeyboardInput();
    thread* mpKeyboardInputThread = new thread(&KeyboardInput::Run, mpkeyboardinput);


    // ROS节点初始化
    ros::init(argc, argv, "dashgo_control");
    // 创建节点句柄
    ros::NodeHandle node;
    // 创建一个Publisher，发布海龟速度指令，让海龟圆周运动
    ros::Publisher vel_pub = node.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);
    tf::TransformListener listener;

    // 设置发布循环的频率,10HZ
    ros::Rate rate(10);
    // 初始化std_msgs::String类型的消息

    string yaml_file = std::string(DATA_DIR) + "config.yaml";
    cv::FileStorage fSettings(yaml_file, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return 0;
    }

    double v = 0, w = 0;
    const double v_step = fSettings["v_step"];
    const double w_step = fSettings["w_step"];
    const double v_max = fSettings["v_max"];
    const double w_max = fSettings["w_max"];
    const double decay_factor = fSettings["decay_factor"];
    cout << "load config file...\n";
    cout << "v_step = " << v_step << "\n";
    cout << "w_step = " << w_step << "\n";
    cout << "v_max = " << v_max << "\n";
    cout << "w_max = " << w_max << "\n";
    cout << "decay_factor = " << decay_factor << "\n";
    geometry_msgs::Twist twist;
    twist.linear.x=v;
    twist.linear.y=0;
    twist.linear.z=0;
    twist.angular.x=0;
    twist.angular.y=0;
    twist.angular.z=w;
    bool quit = false;
    while (ros::ok() && !quit)
    {
        int key = mpkeyboardinput->GetKey();
        switch(key)
        {
          case 'W':
          case 'w': v += v_step;break;
          case 'S':
          case 's': v -= v_step;break;
          case 'A':
          case 'a': w += w_step;break;
          case 'D':
          case 'd': w -= w_step;break;
          case 'Q':
          case 'q': quit = true;break;
          case 'E':
          case 'e': v = 0;w = 0;break;
          default:
          {
            if(v > 0)
            {
                v = std::max(v - decay_factor*v_step, 0.0);
            }
            else if (v < 0) {
                v = std::min(v + decay_factor*v_step, 0.0);
            }
            if(w > 0)
            {
                w = std::max(w - decay_factor*w_step, 0.0);
            }
            else if (w < 0) {
                w = std::min(w + decay_factor*w_step, 0.0);
            }
          }
        }

        v = std::max(std::min(v_max, v), -v_max);
        w = std::max(std::min(w_max, w), -w_max);

        twist.linear.x=v;
        twist.angular.z=w;

        // 发布消息
        vel_pub.publish(twist);
        rate.sleep();
    }
    mpKeyboardInputThread->join();

    delete mpkeyboardinput;
    delete mpKeyboardInputThread;
    return 0;
}
