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

#include "loop/pose_graph.h"
#include "visualization/drawer.h"


#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
#include <turtlesim/Pose.h>
#include <iostream>

#include "mynteye/api/api.h"
#include "mynteye/types.h"
#include "mynteye/device/device.h"
#include "mynteye/device/utils.h"
#include "mynteye/util/times.h"

using namespace std;
Eigen::Vector3d TbcG, TcbG;//轮速计-左目外参
Eigen::Matrix3d RclcrG, RcrclG;//双目外参
Eigen::Vector3d tclcrG, tcrclG;//双目外参
double focal_length;

PoseGraph* mpPoseGraph = new PoseGraph();
Drawer* mpDrawer = new Drawer();

CameraModel::CameraPtr camera_l, camera_r;

int show_image;

void LoadConfigFile(const std::string &file)
{
    cv::FileStorage fSettings(file, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }

    focal_length = fSettings["focal_length"];
    cout << "focal length: " << focal_length << endl;
    camera_l = CameraModel::CameraFactory::instance()->generateCameraFromYamlFile(file, 'l');
    camera_r = CameraModel::CameraFactory::instance()->generateCameraFromYamlFile(file, 'r');

    mpPoseGraph->SetParameter(fSettings);
    mpDrawer->SetParameter(fSettings);
    std::vector<double> extrinsics_Tbc;
    fSettings["extrinsics_Tbc"] >> extrinsics_Tbc;
    TbcG << extrinsics_Tbc[0], extrinsics_Tbc[1], 0;
    TcbG = InverseTransform2D(TbcG);

    std::vector<double> extrinsics_Tcrcl;
    fSettings["extrinsics_Tcrcl"] >> extrinsics_Tcrcl;

    RcrclG << extrinsics_Tcrcl[0], extrinsics_Tcrcl[1], extrinsics_Tcrcl[2],
              extrinsics_Tcrcl[4], extrinsics_Tcrcl[5], extrinsics_Tcrcl[6],
              extrinsics_Tcrcl[8], extrinsics_Tcrcl[9], extrinsics_Tcrcl[10];
    tcrclG << extrinsics_Tcrcl[3], extrinsics_Tcrcl[7], extrinsics_Tcrcl[11];
    RclcrG = RcrclG.transpose();
    tclcrG = - RclcrG*tcrclG;

    std::cout << "Tbc:" << TbcG.transpose() << std::endl;
    std::cout << "Tcb:" << TcbG.transpose() << std::endl;
    std::cout << "Rcrcl:" << RcrclG << std::endl;
    std::cout << "tcrcl:" << tcrclG.transpose() << std::endl;

    show_image = fSettings["show_image"];
}



int main(int argc, char **argv)
{
    string YamlPath = std::string(DATA_DIR) + "config.yaml";
    cv::FileStorage fSettings(YamlPath, cv::FileStorage::READ);
    LoadConfigFile(YamlPath);

    thread* mpPoseGraphThread = new thread(&PoseGraph::Run3DoF, mpPoseGraph);
    thread* mpDrawerThread = new thread(&Drawer::Run, mpDrawer);

    // ROS节点初始化
    ros::init(argc, argv, "dashgo_localization");
    // 创建节点句柄
    ros::NodeHandle node;
    // 创建一个Publisher，发布DashGo速度指令，让DashGo运动
//    ros::Publisher vel_pub = node.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);
    // 创建一个Publisher，发布localization result
    ros::Publisher pose_pub = node.advertise<geometry_msgs::Twist>("/pose", 1000);
    tf::TransformListener listener;
    // 设置发布循环的频率,10HZ
//    ros::Rate rate(10);
    // 初始化std_msgs::String类型的消息

    listener.waitForTransform("/odom", "/base_footprint", ros::Time(0), ros::Duration(3.0));

    auto GetPose = [&](){
        tf::StampedTransform transform;
        while(1)
        {
          try
          {
            listener.lookupTransform("/odom", "/base_footprint", ros::Time(0), transform);
            break;
          }
          catch (tf::TransformException &ex)
          {
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
            continue;
          }
        }
        tf::Quaternion tf_Q = transform.getRotation();
        Eigen::Quaterniond Q(tf_Q.w(), tf_Q.x(), tf_Q.y(), tf_Q.z());
        double yaw = R2ypr(Q.toRotationMatrix()).x();
        Eigen::Vector3d pos;
        pos << transform.getOrigin().x(), transform.getOrigin().y(), yaw;
        return pos;
    };

    MYNTEYE_USE_NAMESPACE
    auto &&api = API::Create(0, nullptr);
    if (!api) return 0;
    bool ok;
    auto &&request = api->SelectStreamRequest(&ok);
    if(!ok) return 0;

    api->SetOptionValue(Option::FRAME_RATE, 10);
    api->ConfigStreamRequest(request);
    api->Start(Source::ALL);
    std::queue<std::pair<timestamp_t, cv::Mat>> image_datas;
    Eigen::Vector3d incremental_pose;
    Eigen::Vector3d absolute_pose, last_absolute_pose;
    // initial value when start up
    last_absolute_pose = GetPose();// x,y, yaw
    incremental_pose = last_absolute_pose;

    // used by relocalization
    Eigen::Vector3d pub_init_pose, pub_new_pose;
    int relo_count = 0;
    const int pub_relo_max = 2;//3s
    while(1)
    {
        api->WaitForStreams();
        auto &&left_data = api->GetStreamData(Stream::LEFT);
        auto &&right_data = api->GetStreamData(Stream::RIGHT);
        absolute_pose = GetPose();// x,y, yaw
        if(left_data.frame.empty() || right_data.frame.empty()) continue;

        if(show_image == 2)
        {
            cv::Mat concat_img;
            cv::hconcat(left_data.frame, right_data.frame, concat_img);
            cv::imshow("stereo image", concat_img);
        }
        else if(show_image == 1)
        {
            cv::imshow("monocular image", left_data.frame);
        }
        int key = cv::waitKey(1);
        if(key == 'q' || key == 'Q')
          break;

        Eigen::Vector3d d_pose = Transform2D(InverseTransform2D(last_absolute_pose), absolute_pose);
        incremental_pose = Transform2D(incremental_pose, d_pose);

        uchar status = mpPoseGraph->GetSolveState(pub_new_pose);
        if(status == 0 && relo_count == pub_relo_max)// start solving
        {
            mpPoseGraph->PushBuf(left_data.frame, right_data.frame, incremental_pose);
            pub_init_pose = incremental_pose;
            mpPoseGraph->SetSolveState(1);
        }
        else if(status == 2)// finished solving but failed
        {
            mpPoseGraph->SetSolveState(0);
        }
        else if(status == 3)// finished solving and success
        {
            Eigen::Vector3d T12 = Transform2D(InverseTransform2D(pub_init_pose), incremental_pose);
            incremental_pose = Transform2D(pub_new_pose, T12);
            mpPoseGraph->SetSolveState(0);
            relo_count = 0;
        }

        mpDrawer->SetCurPose(incremental_pose);
        last_absolute_pose = absolute_pose;

        geometry_msgs::Twist pub_pose;
        pub_pose.linear.x = incremental_pose.x();
        pub_pose.linear.y = incremental_pose.y();
        pub_pose.linear.z = incremental_pose.z();
        pose_pub.publish(pub_pose);
        std::cout << "pose: " << incremental_pose.transpose() << std::endl;

        if(relo_count < pub_relo_max)
          ++relo_count;
    }

    api->Stop(Source::ALL);

    mpDrawer->SetQuit(true);
    mpPoseGraph->SetQuit(true);
    mpPoseGraphThread->join();
    mpDrawerThread->join();

    return 0;
}
