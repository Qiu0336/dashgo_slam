
#include "keyframe.h"
#define MIN_LOOP_NUM 25


template<typename T>
static void reduceVector(vector<T> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

template<typename T>
static void reduceVector(vector<T> &v, vector<bool> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

struct AreaCircle {
    AreaCircle(const float r) : r2(r*r) {}
    bool operator()(const cv::Point2f &p1, const cv::Point2f &p2) { return (p1 - p2).dot(p1 - p2) < r2; }
    float r2;
};

struct AreaLine {
    AreaLine(const Eigen::Vector3d &_line, const float r) : line(_line), r2(r * r)
    {
        inv_denominator = 1.0 / line.head(2).squaredNorm();
    }
    bool operator()(const cv::Point2f &pt)
    {
        const Eigen::Vector3d p(pt.x, pt.y, 1);
        const float num2 = line.dot(p);
        const float squareDist2 = num2 * num2 * inv_denominator;
        return squareDist2 < r2;
    }
    Eigen::Vector3d line;
    float r2;
    float inv_denominator;
};

struct AreaRect {
    AreaRect(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, const float r)
    {
        left = std::min(p1.x(), p2.x()) - r;
        right = std::max(p1.x(), p2.x()) + r;
        top = std::min(p1.y(), p2.y()) - r;
        bottom = std::max(p1.y(), p2.y()) + r;
    }
    bool operator()(const cv::Point2f &pt)
    {
        if(pt.x > left && pt.x < right && pt.y > top && pt.y < bottom)
            return true;
        return false;
    }
    float left;
    float right;
    float top;
    float bottom;
};

int HammingDist(const cv::Mat &des1, const cv::Mat &des2)
{
    int dist = 0;
    for(int i = 0; i < 32; i++)
    {
        const std::bitset<8> &a = des1.at<uchar>(i);
        const std::bitset<8> &b = des2.at<uchar>(i);
        const std::bitset<8> c = a ^ b;
        dist += c.count();
    }
    return dist;
}

// 回环帧的相邻帧匹配
// 全局匹配

void MatchTwoFrameSpecialEpline(const vector<cv::Point2f> &kpt_query,
                                const vector<cv::Point2f> &kpt_train,
                                const cv::Mat &des_query,
                                const cv::Mat &des_train,
                                vector<cv::DMatch> &matches, const float radius)
{
    matches.resize(kpt_query.size());
    int matches_num = 0;
    const Eigen::Matrix3d E21 = Skew(tclcrG)*RclcrG;
    for(int i = 0; i < kpt_query.size(); ++i)
    {
        cv::DMatch best_match(i, 0, 256);
        cv::DMatch second_best_match(i, 0, 256);
        const cv::Point2f &kpt1 = kpt_query[i];

        const Eigen::Vector3d p1(kpt1.x, kpt1.y, 1);
        const Eigen::Vector3d ep_line = E21 * p1;

        AreaLine InLine(ep_line, radius);

        const cv::Mat &des1 = des_query.row(i);
        for(int j = 0; j < des_train.rows; ++j)
        {
            const cv::Point2f &kpt2 = kpt_train[j];
            if(!InLine(kpt2)) continue;

            const cv::Mat &des2 = des_train.row(j);
            int dist = HammingDist(des1, des2);
            if(dist < best_match.distance)
            {
                second_best_match = best_match;
                best_match.distance = dist;
                best_match.trainIdx = j;
            }
            else if(dist < second_best_match.distance)
            {
                second_best_match.distance = dist;
                second_best_match.trainIdx = j;
            }
        }
        if(best_match.distance < 80 &&
                best_match.distance < 0.9 * second_best_match.distance)
            matches[matches_num++] = best_match;
    }
    matches.resize(matches_num);
}

// 按圆匹配,给定candidates
void MatchTwoFrameInCircle(const shared_ptr<KeyFrame> &kf_query,
                                      const shared_ptr<KeyFrame> &kf_train,
                                      vector<cv::DMatch> &matches, const float radius,
                                      const vector<int> &candidates)
{
    auto kpt_query = kf_query->GetKeypoints();
    auto kpt_train = kf_train->GetKeypoints();
    auto des_query = kf_query->GetDescriptors();
    auto des_train = kf_train->GetDescriptors();

    matches.resize(candidates.size());
    int matches_num = 0;
    AreaCircle InCircle(radius);

    for(auto i : candidates)
    {
        cv::DMatch best_match(i, 0, 256);
        cv::DMatch second_best_match(i, 0, 256);
        const cv::KeyPoint &kpt1 = kpt_query[i];
        const cv::Mat &des1 = des_query.row(i);
        for(int j = 0; j < des_train.rows; ++j)
        {
            const cv::KeyPoint &kpt2 = kpt_train[j];
            if(!InCircle(kpt1.pt, kpt2.pt)) continue;

            const cv::Mat &des2 = des_train.row(j);
            int dist = HammingDist(des1, des2);
            if(dist < best_match.distance)
            {
                second_best_match = best_match;
                best_match.distance = dist;
                best_match.trainIdx = j;
            }
            else if(dist < second_best_match.distance)
            {
                second_best_match.distance = dist;
                second_best_match.trainIdx = j;
            }
        }
        if(best_match.distance < 80
                && best_match.distance < 0.9 * second_best_match.distance)
            matches[matches_num++] = best_match;
    }
    matches.resize(matches_num);
}

void MatchTwoFrameInCircle(const shared_ptr<KeyFrame> &kf_query,
                           const shared_ptr<KeyFrame> &kf_train,
                           vector<cv::DMatch> &matches, const float radius)
{
    vector<int> candidates;
    candidates.resize(kf_query->GetKeypoints_norm().size());
    std::iota(candidates.begin(), candidates.end(), 0);
    MatchTwoFrameInCircle(kf_query, kf_train, matches, radius, candidates);
}


// return p2 = T21*p1
Eigen::Vector3d Transform2D(Eigen::Vector3d T21, Eigen::Vector3d p1)
{
    Eigen::Vector3d p2;
    Eigen::Matrix2d R21;
    double theta = M_PI*T21.z()/180.0;
    R21 << cos(theta), -sin(theta), sin(theta), cos(theta);
    p2.head<2>() = R21*p1.head<2>() + T21.head<2>();
    p2.z() = p1.z() + T21.z();
    return p2;
}

// return T12
Eigen::Vector3d InverseTransform2D(Eigen::Vector3d T21)
{
    double theta = M_PI*T21.z()/180.0;
    Eigen::Matrix2d R21;
    R21 << cos(theta), -sin(theta), sin(theta), cos(theta);

    Eigen::Vector3d T12;
    T12.head<2>() = -R21.transpose()*T21.head<2>();
    T12.z() = -T21.z();
    return T12;
}

void Draw_Matches(const cv::Mat &img1, const cv::Mat &img2,
                  const vector<cv::KeyPoint>& kpts1, const vector<cv::KeyPoint>& kpts2,
                  vector<cv::DMatch>& matches, int save_id)
{
    const int width = img1.cols;
    const int height = img1.rows;
    cv::Mat pre_img_clr, cur_img_clr;
    cv::cvtColor(img1, pre_img_clr, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img2, cur_img_clr, cv::COLOR_GRAY2RGB);
    int contact_width = 2*width;
    int contact_height = height;
    cv::Mat contactImg = cv::Mat(contact_height, contact_width, pre_img_clr.type(), cv::Scalar::all(0));
    cv::Mat ROI_1 = contactImg(cv::Rect(0, 0, width, height));
    cv::Mat ROI_2 = contactImg(cv::Rect(width, 0, width, height));
    pre_img_clr.copyTo(ROI_1);
    cur_img_clr.copyTo(ROI_2);

    cv::Mat matchImg = contactImg.clone();

    for(auto m : matches)
    {
        cv::Point2f pt1 = kpts1[m.queryIdx].pt;
        cv::Point2f pt2 = kpts2[m.trainIdx].pt + cv::Point2f(width, 0);
        cv::Scalar color = cv::Scalar(0, 255, 0);
        cv::circle(matchImg, pt1, 2, color, -1);
        cv::circle(matchImg, pt2, 2, color, -1);
        cv::line(matchImg, pt1, pt2, color, 1);
    }
    if(save_id >= 0)
    {
        std::string save_path = std::string(DATA_DIR) + "matches/match_" + std::to_string(save_id) + ".png";
        cv::imwrite(save_path, matchImg);

        std::string save_path2 = std::string(DATA_DIR) + "matches/match2_" + std::to_string(save_id) + ".png";
        cv::imwrite(save_path2, contactImg);
    }
    cv::imshow("matches", matchImg);
    cv::waitKey(1);
}


KeyFrame::KeyFrame(int _index, Eigen::Vector3d& _pose, const vector<cv::KeyPoint>& _keypoints,
                   const vector<cv::Point2f>& _keypoints_norm, const vector<float>& _keypoints_depth,
                   const cv::Mat& _descriptors)
{
    index = _index;
    pose = _pose;
    pose_camera = Transform2D(pose, TbcG);
    keypoints = _keypoints;
    keypoints_norm = _keypoints_norm;
    keypoints_depth = _keypoints_depth;
    descriptors = _descriptors;
}


KeyFrame::KeyFrame(int _index, Eigen::Vector3d &_pose, cv::Mat &_left_image, cv::Mat &_right_image)
{
    index = _index;
    pose = _pose;
    pose_camera = Transform2D(pose, TbcG);

    left_image = _left_image.clone();
    right_image = _right_image.clone();
    loop_index = -1;
    ComputeBRIEFPoint();// 重新提取FAST角点用于回环检测，保存在brief_descriptors
    left_image.release();
    right_image.release();
}

void KeyFrame::ComputeBRIEFPoint()
{
    ORB_Extractor ext(500, 1.2f, 8, 20, 7);
    ext.Detect(left_image, keypoints, descriptors);
    keypoints_norm.resize(keypoints.size());
    keypoints_depth.resize(keypoints.size(), -1);
    for (int i = 0; i < (int)keypoints.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        camera_l->pixel2hnorm(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        keypoints_norm[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }
    vector<cv::KeyPoint> keypoints_r;
    vector<cv::Point2f> keypoints_norm_r;
    cv::Mat descriptors_r;// keypoints的描述子
    ext.Detect(right_image, keypoints_r, descriptors_r);

    keypoints_norm_r.resize(keypoints_r.size());
    for (int i = 0; i < (int)keypoints_r.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        camera_r->pixel2hnorm(Eigen::Vector2d(keypoints_r[i].pt.x, keypoints_r[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        keypoints_norm_r[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }
    vector<cv::DMatch> matches;
    MatchTwoFrameSpecialEpline(keypoints_norm, keypoints_norm_r,
                               descriptors, descriptors_r, matches, 5.0/focal_length);

    vector<Eigen::Vector2d> matched_norm1, matched_norm2;
    matched_norm1.resize(matches.size());
    matched_norm2.resize(matches.size());
    for(int i = 0, iend = matches.size(); i < iend; i++)
    {
        cv::Point2f &p1 = keypoints_norm[matches[i].queryIdx];
        cv::Point2f &p2 = keypoints_norm_r[matches[i].trainIdx];
        matched_norm1[i] = Eigen::Vector2d(p1.x, p1.y);
        matched_norm2[i] = Eigen::Vector2d(p2.x, p2.y);
    }
    FundamentalEstimator fundamentalestimator(0.6, 3/focal_length, 0.99);
    vector<bool> status_F(matches.size(), false);
    bool F_solve_flag = fundamentalestimator.FundamentalRansac(matched_norm1, matched_norm2, status_F);

    reduceVector(matches, status_F);

//    cv::Mat match_img;
//    cv::drawMatches(left_image, keypoints, right_image, keypoints_r, matches, match_img);
//    cv::imshow("match_img", match_img);
//    cv::waitKey(1);

    for(int i = 0; i < matches.size(); ++i)
    {
        cv::Point2f &p1 = keypoints_norm[matches[i].queryIdx];
        cv::Point2f &p2 = keypoints_norm_r[matches[i].trainIdx];
        Eigen::Vector3d xl, xr;
        xl << p1.x, p1.y, 1.0;
        xr << p2.x, p2.y, 1.0;

        // 这里三角化是将滑窗内所有观测到的帧都考虑在内
        Eigen::Matrix4d svd_A;
        Eigen::Matrix<double, 3, 4> P;
        P.leftCols(3).setIdentity();
        P.rightCols(1).setZero();
        Eigen::Vector3d f = xl.normalized();// 这里是整成单位向量来三角化，这样有什么好处呢？？？
        svd_A.row(0) = f[0]*P.row(2) - f[2]*P.row(0);
        svd_A.row(1) = f[1]*P.row(2) - f[2]*P.row(1);

        Eigen::Matrix<double, 3, 4> P2;
        P2.leftCols(3) = RcrclG;
        P2.rightCols(1) = tcrclG;
        Eigen::Vector3d f2 = xr.normalized();// 这里是整成单位向量来三角化，这样有什么好处呢？？？
        svd_A.row(2) = f2[0]*P2.row(2) - f2[2]*P2.row(0);
        svd_A.row(3) = f2[1]*P2.row(2) - f2[2]*P2.row(1);
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols(1);
        float dep = svd_V[2] / svd_V[3];
        if(dep > 0.1 && dep < 20)
        {
            keypoints_depth[matches[i].queryIdx] = dep;
        }
    }
}


void KeyFrame::SetPoseUpdate(const Eigen::Vector3d &_Twb)
{
    pose_update = _Twb;
    pose_update_camera = Transform2D(pose_update, TbcG);
}

void KeyFrame::SetLoopMessage(const Eigen::Vector3d &_loop_pose_ij, const int loop_id)
{
    loop_pose_ij = _loop_pose_ij;// i为回环帧，j为当前帧
    loop_index = loop_id;
}
