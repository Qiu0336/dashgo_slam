
#include "pose_graph.h"

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

PoseGraph::PoseGraph()
{
    keyframe_id = 0;

    SetQuit(false);
    SetSolveState(0);
}

void PoseGraph::SetParameter(cv::FileStorage fSettings)// 加载词汇库
{
    save_path = std::string(fSettings["map_save_path"]);
    voc = new DBoW3::Vocabulary(string(DATA_DIR) + "/dashgo_localization/DBow3/orbvoc.dbow3");
    db.setVocabulary(*voc, false);

    cout << "loading map ..." << endl;
    db.load(save_path + "map.gz");

    for(int i = 0; i < db.size(); ++i)
    {
        int idx = i;
        Eigen::Vector3d pose;
        std::ifstream input(save_path + to_string(idx) + ".txt");
        if(!input.is_open())
            continue;

        std::string line;
        std::getline(input, line);
        std::istringstream iss_pose(line);
        iss_pose >> pose.x() >> pose.y() >> pose.z();

        vector<cv::KeyPoint> keypoints;
        vector<cv::Point2f> keypoints_norm;
        vector<float> keypoints_depth;

        while(std::getline(input, line))//getline得到的字符串，一行中不同数据以","分开
        {
            cv::KeyPoint kpt;
            cv::Point2f pt_norm;
            float depth;
            std::istringstream iss(line);
            iss >> kpt.pt.x >> kpt.pt.y >> pt_norm.x >> pt_norm.y >> depth;
            keypoints.push_back(kpt);
            keypoints_norm.push_back(pt_norm);
            keypoints_depth.push_back(depth);
        }

        cv::Mat descriptors;
        descriptors = cv::imread(save_path + to_string(idx) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        shared_ptr<KeyFrame> keyframe = make_shared<KeyFrame>(keyframe_id, pose, keypoints, keypoints_norm,
                                                              keypoints_depth, descriptors);
        keyframe->SetPoseUpdate(pose);
        keyframelist.push_back(keyframe);
    }
}
void PoseGraph::Run3DoF()
{
    bool flag_detect_loop = true;

    while(!GetQuit())
    {
        chrono::milliseconds dura(5);
        this_thread::sleep_for(dura);

        cv::Mat left_image_msg;
        cv::Mat right_image_msg;
        Eigen::Vector3d pose_msg;
        bool getting_data = false;

        m_buf.lock();
        if(!pose3dof_buf.empty())// 如果buf非空
        {
            left_image_msg = left_image_buf.front();
            right_image_msg = right_image_buf.front();
            pose_msg = pose3dof_buf.front();
            left_image_buf.pop();
            right_image_buf.pop();
            pose3dof_buf.pop();
            getting_data = true;
        }
        m_buf.unlock();
        if(getting_data)
        {
            Eigen::Vector3d relo_pose;
            bool solve_state = false;
            // 创建位姿图中的关键帧, frame_id = -1
            shared_ptr<KeyFrame> keyframe = make_shared<KeyFrame>(-1, pose_msg, left_image_msg, right_image_msg);
            if(flag_detect_loop)
            {
                int loop_index = DetectLoop(keyframe);// 回环检测，这里的index是在pose_graph中新的index

                cout << "loop_index:" << loop_index << endl;
                if(loop_index != -1)// loop_index != -1，说明检测到回环帧了
                {
                    cout << "find loop: " << loop_index << endl;
                    solve_state = FindConnection(keyframe, loop_index, relo_pose);
                }
            }
            if(!solve_state)
                SetSolveState(2);
            else
                SetSolveState(3, relo_pose);
        }
    }
}

int PoseGraph::DetectLoop(shared_ptr<KeyFrame> keyframe)
{
    DBoW3::QueryResults ret;
    db.query(keyframe->GetDescriptors(), ret, 5);// 在原库中查询一遍

    const float threshold = 0.03;
    if(ret.size() >= 1 && ret[0].Score > threshold)
    {
        return ret[0].Id;
    }
    else
    {
        return -1;
    }
}

bool PoseGraph::FindConnection(shared_ptr<KeyFrame> &kf_cur, const int loop_id, Eigen::Vector3d &relo_pose)
{
    shared_ptr<KeyFrame> kf_old = GetKeyFrame(loop_id);// 得到回环帧

    vector<int> solved_candidates_old;
    for(int i = 0; i < kf_old->keypoints_depth.size(); ++i)
    {
        if(kf_old->keypoints_depth[i] > 0)
            solved_candidates_old.push_back(i);
    }
    vector<cv::DMatch> matches;
    MatchTwoFrameInCircle(kf_old, kf_cur, matches, 150, solved_candidates_old);

    if(matches.size() < 20)
    {
        cout << "failed1" << endl;
        return false;
    }
    vector<Eigen::Vector2d> matched_norm1, matched_norm2;
    matched_norm1.resize(matches.size());
    matched_norm2.resize(matches.size());
    auto keypoints_norm1 = kf_old->GetKeypoints_norm();
    auto keypoints_norm2 = kf_cur->GetKeypoints_norm();
    for(int i = 0, iend = matches.size(); i < iend; i++)
    {
        cv::Point2f &p1 = keypoints_norm1[matches[i].queryIdx];
        cv::Point2f &p2 = keypoints_norm2[matches[i].trainIdx];
        matched_norm1[i] = Eigen::Vector2d(p1.x, p1.y);
        matched_norm2[i] = Eigen::Vector2d(p2.x, p2.y);
    }

    FundamentalEstimator fundamentalestimator(0.6, 3/focal_length, 0.99);
    vector<bool> status_F(matches.size(), false);
    bool F_solve_flag = fundamentalestimator.FundamentalRansac(matched_norm1, matched_norm2, status_F);

    reduceVector(matches, status_F);

    if(matches.size() < 20 || !F_solve_flag)
    {
        cout << "failed2" << endl;
        return false;
    }


    vector<PnPMatch> pnp_matches;

    Eigen::Matrix3d Rwc_new;
    Eigen::Vector3d twc_new;
    Rwc_new.setIdentity();
    twc_new.setZero();

    // 求解PnP需要Rcw和tcw，故求逆
    Eigen::Matrix3d Rcw_new = Rwc_new.transpose();
    Eigen::Vector3d tcw_new = - Rcw_new*twc_new;

    for(auto &m : matches)
    {
        float depth = kf_old->keypoints_depth[m.queryIdx];
        auto &p1_ = keypoints_norm1[m.queryIdx];
        auto &p2_ = keypoints_norm2[m.trainIdx];
        Eigen::Vector3d p1;
        p1 << p1_.x, p1_.y, 1;
        Eigen::Vector3d world_point = p1*depth;
        pnp_matches.push_back(PnPMatch(world_point, Eigen::Vector2d(p2_.x, p2_.y)));
    }

    if(pnp_matches.size() < 20)
    {
        cout << "failed3" << endl;
        return false;
    }

    PnPEstimator pnpestimator(0.6, 3/focal_length, 0.99, 20);
    vector<bool> status2(matches.size(), false);
    bool solve_flag = pnpestimator.PnPRansac(pnp_matches, Rcw_new, tcw_new, status2);

    if(solve_flag)
    {
      Rwc_new = Rcw_new.transpose();
      twc_new = - Rwc_new*tcw_new;
      Eigen::Vector3d ypr_new = R2ypr(Rwc_new);
      Eigen::Vector3d Tc1c2_new;
      Tc1c2_new << twc_new.z(), -twc_new.x(), -ypr_new.y();

      Eigen::Vector3d Twc1 = kf_old->GetPoseCamera();
      relo_pose = Transform2D(Transform2D(Twc1, Tc1c2_new), TcbG);
      cout << "successfully relocalized !!!" <<endl;
      return true;
    }
    else
    {
        cout << "solve failed" << endl;
        return false;
    }
}
// 通过index获取keyframe
shared_ptr<KeyFrame> PoseGraph::GetKeyFrame(const int index)
{
    if(index < 0 || index >= (int)keyframelist.size())
        return nullptr;
    else
        return *(keyframelist.begin() + index);
}

void PoseGraph::PushBuf(const cv::Mat& left_img, const cv::Mat& right_img, const Eigen::Vector3d& Pose)
{
    unique_lock<mutex> lock(m_buf);
    left_image_buf.push(left_img);
    right_image_buf.push(right_img);
    pose3dof_buf.push(Pose);
}

void PoseGraph::SetQuit(bool x)
{
    unique_lock<mutex> lock(m_quit);
    quit_flag = x;
}

bool PoseGraph::GetQuit()
{
    unique_lock<mutex> lock(m_quit);
    return quit_flag;
}

void PoseGraph::SetSolveState(uchar x, Eigen::Vector3d relo_pose)
{
    unique_lock<mutex> lock(m_solve_state);
    solve_state = x;
    relocalization_pose = relo_pose;
}

uchar PoseGraph::GetSolveState(Eigen::Vector3d& relo_pose)
{
    unique_lock<mutex> lock(m_solve_state);
    relo_pose = relocalization_pose;
    return solve_state;
}

vector<shared_ptr<KeyFrame>> PoseGraph::GetKeyframelist()
{
    unique_lock<mutex> lock(m_keyframelist);
    return keyframelist;
}
