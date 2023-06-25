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

#include "DBow3/src/DBoW3.h"
#include "DBow3/src/DescManip.h"

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
    void SetParameter(cv::FileStorage fSettings);

    void Run3DoF();
    int DetectLoop(shared_ptr<KeyFrame> keyframe);
    bool FindConnection(shared_ptr<KeyFrame> &cur_kf, const int loop_id, Eigen::Vector3d &relo_pose);
    shared_ptr<KeyFrame> GetKeyFrame(const int index);

    void PushBuf(const cv::Mat& left_img, const cv::Mat& right_img, const Eigen::Vector3d& Pose);

    void SetQuit(bool x);
    bool GetQuit();

    void TryToGetPose(bool x);
    void SetSolveState(uchar x, Eigen::Vector3d relo_pose = Eigen::Vector3d::Zero());
    uchar GetSolveState(Eigen::Vector3d& relo_pose);
    vector<shared_ptr<KeyFrame>> GetKeyframelist();

    mutex m_keyframelist;
    vector<shared_ptr<KeyFrame>> keyframelist;// 保存了posegraph中所有的关键帧

    mutex m_buf;
    queue<cv::Mat> left_image_buf;
    queue<cv::Mat> right_image_buf;
    queue<Eigen::Vector3d> pose3dof_buf;

    mutex m_quit;
    bool quit_flag;// 是否序列已经结束，如果结束就跳出循环并保存回环位姿

    mutex m_solve_state;
    uchar solve_state;// 0:空闲，1:正在计算，2:算完，失败，3:算完，成功
    Eigen::Vector3d relocalization_pose;

    int keyframe_id;

    string save_path;
    DBoW3::Vocabulary* voc;
    DBoW3::Database db;
};
