
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
    earliest_loop_index = -1;
    last_loop = -1;
    keyframe_id = 0;
    pose_drift.setZero();

    SetQuit(false);
}

void PoseGraph::Run3DoF()
{
    int skip_first_cnt = 10;

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
            if (skip_first_cnt < 10)//跳过最开始的几帧，不处理
            {
                skip_first_cnt++;
                continue;
            }

            // 创建位姿图中的关键帧
            shared_ptr<KeyFrame> keyframe = make_shared<KeyFrame>(keyframe_id, pose_msg, left_image_msg, right_image_msg);

            AddKeyFrame3DoF(keyframe, true);
            keyframe_id ++;
        }
    }
}



void PoseGraph::AddKeyFrame3DoF(shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop)
{
    Eigen::Vector3d cur_pose, cur_pose_update;

    cur_pose = cur_kf->GetPose();

    m_keyframelist.lock();
    keyframelist.push_back(cur_kf);
    m_keyframelist.unlock();

    // solve
    {
        int cur_id = cur_kf->GetIndex();
        if(cur_id >= 1)
        {
            if(FindConnection(cur_kf, cur_id - 1))
            {
                Eigen::Vector3d pose1 = GetKeyFrame(cur_id - 1)->GetPose();
                Eigen::Vector3d pose2 = cur_kf->GetPose();
                Eigen::Vector3d loop_pose_bibj = Transform2D(InverseTransform2D(pose1), pose2);
                double dtheta = loop_pose_cicj.z()*M_PI / 180.0;
                Eigen::Matrix2d Rcicj;
                Rcicj << cos(dtheta), -sin(dtheta),
                    sin(dtheta), cos(dtheta);
//                Eigen::Vector2d tcb = (Rcicj - Eigen::Matrix2d::Identity()).inverse()*(loop_pose_bibj.head<2>() - loop_pose_cicj.head<2>());
//                Eigen::Vector2d tbc = -tcb;
                Eigen::Matrix2d A = Rcicj - Eigen::Matrix2d::Identity();
                Eigen::Vector2d b = loop_pose_bibj.head<2>() - loop_pose_cicj.head<2>();
                vec_A.push_back(A);
                vec_b.push_back(b);

                if(vec_A.size() == calib_frames)
                {
                    Eigen::MatrixXd H;
                    H.resize(2*calib_frames, 2);
                    Eigen::VectorXd B;
                    B.resize(2*calib_frames);
                    for(int i = 0; i < calib_frames; ++i)
                    {
                        H.block<2, 2>(2*i, 0) = vec_A[i];
                        B.segment<2>(2*i) = vec_b[i];
                    }
                    Eigen::Vector2d tcb = (H.transpose()*H).ldlt().solve(H.transpose()*B);
                    Eigen::Vector2d tbc = -tcb;
                    cout << "result: " << tbc.transpose() << endl;
                    vec_A.clear();
                    vec_b.clear();
                }
            }
        }

    }
}

bool PoseGraph::FindConnection(shared_ptr<KeyFrame> &kf_cur, const int loop_id)
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


//    Draw_Matches(kf_old->GetImage(), kf_cur->GetImage(),
//                 kf_old->GetKeypoints(), kf_cur->GetKeypoints(),
//                 matches, kf_cur->GetIndex());

    vector<PnPMatch> pnp_matches;

    Eigen::Matrix3d Rwc_new, Rwc1;
    Eigen::Vector3d twc_new, twc1;
    Rwc1.setIdentity();
    twc1.setZero();
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
        Eigen::Vector3d world_point = Rwc1*p1*depth + twc1;
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

        Eigen::Vector3d Twc_new;
        Twc_new << twc_new.z(), -twc_new.x(), -ypr_new.y();
        loop_pose_cicj = Twc_new;
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

vector<shared_ptr<KeyFrame>> PoseGraph::GetKeyframelist()
{
    unique_lock<mutex> lock(m_keyframelist);
    return keyframelist;
}
