
#pragma once

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include "loop/pose_graph.h"

using namespace std;
using namespace Eigen;

void DrawNone();

extern PoseGraph* mpPoseGraph;

class Drawer
{
    public:
    Drawer();
    void SetParameter(cv::FileStorage fSettings);
    void DrawBackground();
    void DrawCamera2D();
    void DrawTrajectory2D();

    void Run();
    void SetCurPose(const Eigen::Vector3d& cur_pose_2d_);
    Eigen::Vector3d GetCurPose();
    void SetQuit(bool x);
    bool GetQuit();

    float mBackgroundpatchsize;// 背景一块的大小
    int mBackgroundpatchcount;// 背景的块数
    float mPointSize;
    float mCameraSize;// 相机的大小
    float mCameraLineWidth;// 相机的线粗

    mutex m_cur_pose_2d;
    Eigen::Vector3d cur_pose_2d;

    mutex m_quit;
    bool quit_flag;
};

