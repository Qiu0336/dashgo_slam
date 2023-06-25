#pragma once

#include <iostream>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <sstream>
#include <mutex>
#include <queue>
#include <thread>
#include <unistd.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "util/io.h"
#include "util/timer.h"
#include "keyframe.h"
#include "ceresloop.h"
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "pnp.h"

using namespace std;
using namespace io;

class MapPoint
{
    public:
    MapPoint(){}
    bool solved{false};
    int id;// 表示该点在loop帧的keypoint的index
    Eigen::Vector3d world_point;
    map<int, int> loop_observations;
    map<int, int> cur_observations;
};

class PoseGraph
{
    public:
    PoseGraph();
    void Run3DoF();
    void AddKeyFrame3DoF(shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop);
    bool FindConnection(shared_ptr<KeyFrame> &cur_kf, const int loop_id);
    shared_ptr<KeyFrame> GetKeyFrame(const int index);

    void PushBuf(const cv::Mat& left_img, const cv::Mat& right_img, const Eigen::Vector3d& Pose);

    void SetQuit(bool x);
    bool GetQuit();
    vector<shared_ptr<KeyFrame>> GetKeyframelist();

    int earliest_loop_index;// 初始值为-1

    mutex m_keyframelist;
    vector<shared_ptr<KeyFrame>> keyframelist;// 保存了posegraph中所有的关键帧

    mutex m_buf;
    queue<cv::Mat> left_image_buf;
    queue<cv::Mat> right_image_buf;
    queue<Eigen::Vector3d> pose3dof_buf;

    Eigen::Vector3d pose_drift;

    mutex m_quit;
    bool quit_flag;// 是否序列已经结束，如果结束就跳出循环并保存回环位姿

    Eigen::Vector3d loop_pose_cicj;

    vector<Eigen::Matrix2d> vec_A;
    vector<Eigen::Vector2d> vec_b;

    int keyframe_id;
    int last_loop;// 上一次检测到回环的帧id
};
