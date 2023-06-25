#include <algorithm>
#include <cmath>
// STL
#include <iterator>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
#include <turtlesim/Pose.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "util/so3.h"
#include "util/timer.h"

using namespace std;

std::mutex m_pose;
Eigen::Vector3d pose_msg;

double translation_error = 0.08;// meter
double angle_error = 5;// degree

void ROS_Spin()
{
    ros::spin();
}

void ChatterCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
    m_pose.lock();
    pose_msg.x() = msg->linear.x;
    pose_msg.y() = msg->linear.y;
    pose_msg.z() = msg->linear.z;
    m_pose.unlock();
}

Eigen::Vector3d GetPose()
{
    Eigen::Vector3d pose;
    m_pose.lock();
    pose = pose_msg;
    m_pose.unlock();
    return pose;
}


int main(int argc, char **argv)
{
    string YamlPath = std::string(DATA_DIR) + "config.yaml";
    cv::FileStorage fSettings(YamlPath, cv::FileStorage::READ);

    string Destination_Path = std::string(DATA_DIR) + "destination.yaml";
    cv::FileStorage fSettings2(Destination_Path, cv::FileStorage::READ);

    vector<Eigen::Vector2d> dst_positions;
    for(int i = 0; i < 13; ++i)
    {
        std::vector<double> dst_pos;
        string dst_str = "dst_" + to_string(i);
        fSettings2[dst_str] >> dst_pos;
        dst_positions.emplace_back(dst_pos[0], dst_pos[1]);
        std::cout << "dts_" << i << " : " << dst_pos[0] << ", " << dst_pos[1] << endl;
    }

    // ROS节点初始化
    ros::init(argc, argv, "dashgo_navigation");
    // 创建节点句柄
    ros::NodeHandle node;
    // 创建一个Subscriber，Subscribe pose
    ros::Subscriber pose_sub = node.subscribe<geometry_msgs::Twist>("/pose", 1000, ChatterCallback);
    // 创建一个Publisher，发布DashGo速度指令，让DashGo运动
    ros::Publisher vel_pub = node.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);

    tf::TransformListener listener;
    // 设置发布循环的频率,10HZ
    ros::Rate rate(10);
    // 初始化std_msgs::String类型的消息

    listener.waitForTransform("/odom", "/base_footprint", ros::Time(0), ros::Duration(3.0));

    auto TurnToAbsoluteAngle = [&](double dst_angle){
        geometry_msgs::Twist twist;
        twist.linear.x=0;
        twist.linear.y=0;
        twist.linear.z=0;
        twist.angular.x=0;
        twist.angular.y=0;
        twist.angular.z=0;
        double w_min = 0.1;
        double w_max = 0.4;
        double w_acc = M_PI/3;
        double angle_th = (w_max*w_max - w_min*w_min)/(2*w_acc);

        double cur_angle = GetPose().z();
        double initial_diff_angle = NormalizeAngle(dst_angle - cur_angle);
        double diff_angle = initial_diff_angle;
        while(fabs(diff_angle) > angle_error)
        {
            double min_th = max(M_PI*min(fabs(diff_angle), fabs(initial_diff_angle) - fabs(diff_angle))/180.0, 0.0);
            double cur_w = min_th < angle_th? std::sqrt(2*w_acc*min_th + w_min*w_min) : w_max;
            twist.angular.z = diff_angle>0? cur_w : -cur_w;
            vel_pub.publish(twist);
            rate.sleep();
            cur_angle = GetPose().z();
            diff_angle = NormalizeAngle(dst_angle - cur_angle);
        }
    };

    auto GoToAbsolutePosition = [&](Eigen::Vector2d dst_pos){
        geometry_msgs::Twist twist;
        twist.linear.x=0;
        twist.linear.y=0;
        twist.linear.z=0;
        twist.angular.x=0;
        twist.angular.y=0;
        twist.angular.z=0;

        double v_min = 0.08;
        double v_max = 0.4;
        double v_acc = 0.5;
        double w_max = 0.4;
        double distance_th = (v_max*v_max - v_min*v_min)/(2*v_acc);
        Eigen::Vector3d cur_pose = GetPose();
        double initial_distance = (dst_pos - cur_pose.head<2>()).norm();
        double cur_distance = initial_distance;
        while(fabs(cur_distance) > translation_error)
        {
            double min_th = max(min(cur_distance, initial_distance - cur_distance), 0.0);
            twist.linear.x = min_th < distance_th? std::sqrt(2*v_acc*min_th + v_min*v_min) : v_max;


            Eigen::Vector2d d_pos = dst_pos - cur_pose.head<2>();
            double dst_angle = 180.0*atan2(d_pos.y(), d_pos.x())/ M_PI;

            twist.angular.z = M_PI*NormalizeAngle(dst_angle - cur_pose.z())/ 180.0;
            twist.angular.z = max(min(twist.angular.z, w_max), -w_max);

            vel_pub.publish(twist);
            rate.sleep();
            cur_pose = GetPose();
            cur_distance = (dst_pos - cur_pose.head<2>()).norm();
        }
    };

    auto GoToDestination = [&](Eigen::Vector2d dst_dst){
        Eigen::Vector2d cur_pos = GetPose().head<2>();
        Eigen::Vector2d d_pos = dst_dst - cur_pos;
        double dst_angle = 180.0*atan2(d_pos.y(), d_pos.x())/M_PI;
        TurnToAbsoluteAngle(dst_angle);
        GoToAbsolutePosition(dst_dst);
    };

    auto GoToOrigin = [&](){
        Eigen::Vector2d cur_pos = GetPose().head<2>();
        Eigen::Vector2d dst_dst = Eigen::Vector2d::Zero();
        Eigen::Vector2d d_pos = dst_dst - cur_pos;
        double dst_angle = 180.0*atan2(d_pos.y(), d_pos.x())/M_PI;
        TurnToAbsoluteAngle(dst_angle);
        GoToAbsolutePosition(dst_dst);
        TurnToAbsoluteAngle(0);
    };




//    auto TurnToAbsoluteAngle = [&](double dst_angle){
//        geometry_msgs::Twist twist;
//        twist.linear.x=0;
//        twist.linear.y=0;
//        twist.linear.z=0;
//        twist.angular.x=0;
//        twist.angular.y=0;
//        twist.angular.z=0;
//        double w_ave = 0.25;

//        double cur_angle = GetPose().z();
//        double initial_diff_angle = NormalizeAngle(dst_angle - cur_angle);
//        double diff_angle = initial_diff_angle;
//        while(fabs(diff_angle) > angle_error)
//        {
//            Timer timer;
//            timer.Start();
//            twist.angular.z = diff_angle>0? w_ave : -w_ave;
//            vel_pub.publish(twist);
//            rate.sleep();
//            cur_angle = GetPose().z();
//            diff_angle = NormalizeAngle(dst_angle - cur_angle);
//            timer.PrintMilliseconds();
//        }
//    };

//    auto GoToAbsolutePosition = [&](Eigen::Vector2d dst_pos){
//        geometry_msgs::Twist twist;
//        twist.linear.x=0;
//        twist.linear.y=0;
//        twist.linear.z=0;
//        twist.angular.x=0;
//        twist.angular.y=0;
//        twist.angular.z=0;

//        double v_ave = 0.2;
//        double w_max = 0.4;
//        Eigen::Vector3d cur_pose = GetPose();
//        double initial_distance = (dst_pos - cur_pose.head<2>()).norm();
//        double cur_distance = initial_distance;
//        while(fabs(cur_distance) > translation_error)
//        {
//            twist.linear.x = v_ave;

//            Eigen::Vector2d d_pos = dst_pos - cur_pose.head<2>();
//            double dst_angle = 180.0*atan2(d_pos.y(), d_pos.x())/ M_PI;

//            twist.angular.z = M_PI*NormalizeAngle(dst_angle - cur_pose.z())/ 180.0;
//            twist.angular.z = max(min(twist.angular.z, w_max), -w_max);

//            vel_pub.publish(twist);
//            rate.sleep();
//            cur_pose = GetPose();
//            cur_distance = (dst_pos - cur_pose.head<2>()).norm();
//        }
//    };

//    auto GoToDestination = [&](Eigen::Vector2d dst_dst){
//        Eigen::Vector2d cur_pos = GetPose().head<2>();
//        Eigen::Vector2d d_pos = dst_dst - cur_pos;
//        double dst_angle = 180.0*atan2(d_pos.y(), d_pos.x())/M_PI;
//        TurnToAbsoluteAngle(dst_angle);
//        GoToAbsolutePosition(dst_dst);
//    };

//    auto GoToOrigin = [&](){
//        Eigen::Vector2d cur_pos = GetPose().head<2>();
//        Eigen::Vector2d dst_dst = Eigen::Vector2d::Zero();
//        Eigen::Vector2d d_pos = dst_dst - cur_pos;
//        double dst_angle = 180.0*atan2(d_pos.y(), d_pos.x())/M_PI;
//        TurnToAbsoluteAngle(dst_angle);
//        GoToAbsolutePosition(dst_dst);
//        TurnToAbsoluteAngle(0);
//    };



    std::thread spin_thread = std::thread(&ROS_Spin);

    bool is_in_pay_region = false;
    while(1)
    {
        Eigen::Vector3d pose = GetPose();
        std::cout << pose.transpose() << std::endl;
        cout << "choose destination:" << endl;
        int key;
        cin >> key;
        if(key >= 0 && key <= 9)
        {
            cout << "go to destination " << key << endl;
            sleep(2);
            if(key != 1)
            {
              if(is_in_pay_region) {
                GoToDestination(dst_positions[12]);
                sleep(1);
                GoToDestination(dst_positions[11]);
                sleep(1);
                GoToDestination(dst_positions[10]);
                sleep(1);
                GoToDestination(dst_positions[0]);
                is_in_pay_region = false;
              }
              GoToDestination(dst_positions[key]);
            }
            else {
              if(!is_in_pay_region) {
                std::cout << "mark0" << std::endl;
                GoToDestination(dst_positions[0]);
                std::cout << "mark1" << std::endl;
                sleep(1);
                GoToDestination(dst_positions[10]);
                std::cout << "mark2" << std::endl;
                sleep(1);
                GoToDestination(dst_positions[11]);
                std::cout << "mark3" << std::endl;
                sleep(1);
                GoToDestination(dst_positions[12]);
                std::cout << "mark4" << std::endl;
                sleep(1);
                GoToDestination(dst_positions[1]);
                is_in_pay_region = true;
              }
            }
        }
    }

    return 0;
}
