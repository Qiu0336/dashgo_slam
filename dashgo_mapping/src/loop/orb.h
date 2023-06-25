#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

class _ExtractorNode {
public:
    _ExtractorNode() : bNoMore(false) {}

    void DivideNode(_ExtractorNode &n1, _ExtractorNode &n2, _ExtractorNode &n3, _ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<_ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORB_Extractor {
public:
    ORB_Extractor(int n_features, float scale_factor, int n_levels, int initial_threshold_fast, int min_threshold_fast);

    void Detect(cv::InputArray _image, vector<cv::KeyPoint> &_keypoints, cv::OutputArray _descriptors);

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(vector<vector<cv::KeyPoint>> &allKeypoints);
    vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys,
                                           const int &minX, const int &maxX, const int &minY,
                                           const int &maxY, const int &N);

    vector<cv::Mat> mvImagePyramid;

    vector<cv::Point> pattern;
    int n_features;
    double scale_factor;
    int n_levels;
    int initial_threshold_fast;
    int min_threshold_fast;
    vector<int> mn_featuresPerLevel;

    vector<int> umax;

    vector<float> mvscale_factor;
    vector<float> mvInvscale_factor;

};


namespace cv
{
void detectAndCompute( InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors);
}
