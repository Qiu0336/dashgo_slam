#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <random>
#include "util/so3.h"

using namespace std;

struct RandomSampler
{
    void RandomShuffle(vector<int> &sample, int sample_size)
    {
        const int kNumElements = static_cast<int>(sample.size());
        for(int i = 0; i < sample_size; ++i)
        {
            std::uniform_int_distribution<int> dist(i, kNumElements - 1);
            // randomly select a value from [i,kNumElements]
             int idx = dist(rng_);
//		int idx = 0;
            // move the selected value to the top
            std::swap(sample[i], sample[idx]);
        }
    }

    // Draws minimal sample.
    void ShuffleSample(int num_samples, int sample_size, vector<int> &random_sample)
    {
        // increasing filling [0, num_samples]
        random_sample.resize(num_samples);
        std::iota(random_sample.begin(), random_sample.end(), 0);

        RandomShuffle(random_sample, sample_size);
        // only reserve the top sample_size index
        random_sample.resize(sample_size);
    }

    std::mt19937 rng_;
};

struct PnPMatch
{
    PnPMatch() {}
    PnPMatch(const Eigen::Vector3d& _point3d, const Eigen::Vector2d& _point2d_norm,
             const Eigen::Matrix3d _Rc_ref = Eigen::Matrix3d::Identity(),
             const Eigen::Vector3d _tc_ref = Eigen::Vector3d::Zero()):
        point3d(_point3d), point2d_norm(_point2d_norm), Rc_ref(_Rc_ref), tc_ref(_tc_ref)
    {}

    Eigen::Vector3d point3d;
    Eigen::Vector2d point2d_norm;
    Eigen::Matrix3d Rc_ref;
    Eigen::Vector3d tc_ref;
};

class PnPEstimator
{
public:
    PnPEstimator(double _min_inlier_ratio = 0.6, double _max_error = 0.02,
                 double _confidence = 0.99, int _sample_size = 20,
                 int _min_iterations = 100, int _max_iterations = 10000)
        : min_inlier_ratio(_min_inlier_ratio), max_error(_max_error), confidence(_confidence),
          sample_size(_sample_size), min_iterations(_min_iterations),
          max_iterations(_max_iterations)
    {}

    double SolvePnP(const vector<PnPMatch> &matches, Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw, bool print = false);

    void SolvePnP4DoF(const vector<PnPMatch> &matches, Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);

    bool PnPRansac(const vector<PnPMatch> &matches,
                   Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw, vector<bool> &inliners);

private:

    int Evaluate(const vector<PnPMatch> &matches,
                 const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw, vector<bool> &inliners);


//    void BundlePnPRansac(vector<Eigen::Vector3d> &points3d, vector<Eigen::Vector2d> &points2d_norm,
//                         Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);
//    void BundlePnPRansac(vector<cv::Point3f> &points3d, vector<cv::Point2f> &points2d_norm,
//                         Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);

    RandomSampler random_sampler;

    double min_inlier_ratio;
    double max_error;
    double confidence;
    int sample_size;
    int min_iterations;
    int max_iterations;
};


class FundamentalEstimator
{
public:
    FundamentalEstimator(double _min_inlier_ratio = 0.6, double _max_error = 5,
                 double _confidence = 0.99, int _sample_size = 8,
                 int _min_iterations = 100, int _max_iterations = 10000)
        : min_inlier_ratio(_min_inlier_ratio), max_error(_max_error*_max_error), confidence(_confidence),
          sample_size(_sample_size), min_iterations(_min_iterations),
          max_iterations(_max_iterations)
    {}

    void SolveFundamental(const vector<Eigen::Vector2d> &points1, const vector<Eigen::Vector2d> &points2,
                          Eigen::Matrix3d &F_matrix);

    bool FundamentalRansac(const vector<Eigen::Vector2d> &points1,
                           const vector<Eigen::Vector2d> &points2, vector<bool> &inliners);

private:

    int Evaluate(const vector<Eigen::Vector2d> &points1,
                 const vector<Eigen::Vector2d> &points2,
                 const Eigen::Matrix3d &F, vector<bool> &inliners);


    RandomSampler random_sampler;

    double min_inlier_ratio;
    double max_error;//Squared [sampson distance]
    double confidence;
    int sample_size;
    int min_iterations;
    int max_iterations;
};

