#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <bitset>

#include "orb.h"
#include "camera/camera.h"
#include "util/io.h"
#include "util/so3.h"
#include "util/timer.h"
#include "pnp.h"

using namespace io;
using namespace std;

extern int calib_frames;
extern CameraModel::CameraPtr camera_l, camera_r;
extern Eigen::Matrix3d RclcrG, RcrclG;
extern Eigen::Vector3d tclcrG, tcrclG;
extern double focal_length;

class KeyFrame
{
    public:
    KeyFrame(int _index, Eigen::Vector3d &_pose, cv::Mat &_left_image, cv::Mat &_right_image);// 3dof
    void ComputeBRIEFPoint();
    const vector<cv::KeyPoint>& GetKeypoints() { return keypoints; }
    const vector<cv::Point2f>& GetKeypoints_norm() { return keypoints_norm; }
    const cv::Mat& GetDescriptors() { return descriptors; }
    const cv::Mat& GetImage() { return left_image; }
    int GetIndex() { return index; }

    Eigen::Vector3d GetPose() { return pose; }
    int index;// 该帧的id
    Eigen::Vector3d pose;

    cv::Mat left_image, right_image;// 该帧对应的图片
    vector<cv::KeyPoint> keypoints;// 该帧提取的FAST特征点
    vector<cv::Point2f> keypoints_norm;// 该帧提取的FAST特征点的归一化平面坐标
    vector<float> keypoints_depth;// 该帧keypoints点的深度，若为-1则为匹配失败
    cv::Mat descriptors;// keypoints的描述子
};

void MatchTwoFrameInCircle(const shared_ptr<KeyFrame> &kf_query,
                           const shared_ptr<KeyFrame> &kf_train,
                           vector<cv::DMatch> &matches, const float radius,
                           const vector<int> &candidates);

void MatchTwoFrameInCircle(const shared_ptr<KeyFrame> &kf_query,
                           const shared_ptr<KeyFrame> &kf_train,
                           vector<cv::DMatch> &matches, const float radius);

Eigen::Vector3d Transform2D(Eigen::Vector3d T21, Eigen::Vector3d p1);
Eigen::Vector3d InverseTransform2D(Eigen::Vector3d T21);
void Draw_Matches(const cv::Mat &img1, const cv::Mat &img2,
                  const vector<cv::KeyPoint>& kpts1, const vector<cv::KeyPoint>& kpts2,
                  vector<cv::DMatch>& matches, int save_id = -1);

