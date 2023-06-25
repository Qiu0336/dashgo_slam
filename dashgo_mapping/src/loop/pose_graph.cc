
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

void PoseGraph::SetParameter(cv::FileStorage fSettings)// 加载词汇库
{
    save_path = std::string(fSettings["map_save_path"]);
    voc = new DBoW3::Vocabulary(string(DATA_DIR) + "/dashgo_mapping/DBow3/orbvoc.dbow3");
    db.setVocabulary(*voc, false);
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
    SaveMap();
}



void PoseGraph::AddKeyFrame3DoF(shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop)
{
    Eigen::Vector3d cur_pose, cur_pose_update;

    cur_pose = cur_kf->GetPose();
    cur_pose_update = DriftRemove(cur_pose);
    cur_kf->SetPoseUpdate(cur_pose_update);

    m_keyframelist.lock();
    keyframelist.push_back(cur_kf);
    m_keyframelist.unlock();

    if(flag_detect_loop)
    {
        int loop_index = DetectLoop(cur_kf);// 回环检测，这里的index是在pose_graph中新的index

        if(loop_index > 0)// loop_index != -1，说明检测到回环帧了
        {
            if(FindConnection(cur_kf, loop_index))// 寻找当前帧和回环帧之间的关联。判断是否是真回环
            {
                m_keyframelist.lock();
                Optimization_Loop_3DoF();
                m_keyframelist.unlock();
            }
        }
    }

    db.add(cur_kf->GetDescriptors());// 再将此帧描述子放入词袋库中
}

int PoseGraph::DetectLoop(shared_ptr<KeyFrame> keyframe)
{
    const int frame_index = keyframe->GetIndex();
    if(frame_index < 30 || frame_index - last_loop < 20)// 与上一次检测到的回环帧至少间隔5帧
        return -1;

    DBoW3::QueryResults ret;
    db.query(keyframe->GetDescriptors(), ret, 10, frame_index - 30);// 在原库中查询一遍

    bool find_loop = false;
    const float threshold = 0.03;
//     判断检测到回环条件：最大分数 > 0.05且前4个结果所有分数均 > 0.015
    if(ret.size() >= 1 && ret[0].Score > threshold)
    {
        find_loop = true;
    }
    if(find_loop)// 如果找到回环并且pose_graph中的关键帧
    {
        int min_index = -1;
        for(unsigned int i = 0; i < ret.size(); i++)
        {
            if(min_index == -1 || (ret[i].Id < min_index && ret[i].Score > threshold))
                min_index = ret[i].Id;
        }
//        if(min_index > 0)
//        {
//            std::string save_path_src = std::string(DATA_DIR) + "loop/" + std::to_string(frame_index) + ".png";
//            cv::imwrite(save_path_src, keyframe->GetImage());
//            for(int i = 0; i < 10; ++i)
//            {
//                std::string save_path = std::string(DATA_DIR) + "loop/" + std::to_string(frame_index) + "_" + std::to_string(i) + ".png";
//                cv::Mat img = GetKeyFrame(ret[i].Id)->GetImage();
//                cv::Mat img_clr;
//                cv::cvtColor(img, img_clr, cv::COLOR_GRAY2RGB);
//                std::string str1 = std::to_string(ret[i].Score);
//                cv::putText(img_clr, str1, cv::Point2f(0, img_clr.rows - 5), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
//                cv::imwrite(save_path, img_clr);
//            }
//        }
        return min_index;// 找出id最小并且回环Score > 0.015的那一帧
    }
    else
        return -1;
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

    vector<PnPMatch> pnp_matches;

    Eigen::Matrix3d Rwc_new, Rwc1;
    Eigen::Vector3d twc_new, twc1;
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
        Eigen::Vector3d loop_pose_bibj = Transform2D(Transform2D(TbcG, Tc1c2_new), TcbG);
        kf_cur->SetLoopMessage(loop_pose_bibj, loop_id);

//        Draw_Matches(kf_old->GetImage(), kf_cur->GetImage(),
//                     kf_old->GetKeypoints(), kf_cur->GetKeypoints(),
//                     matches, kf_cur->GetIndex());

        if(earliest_loop_index > loop_id || earliest_loop_index == -1)
            earliest_loop_index = loop_id;
        last_loop = kf_cur->index;
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


void PoseGraph::Optimization_Loop_3DoF()
{

    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 1;
    options.max_solver_time_in_seconds = 0.5;
    options.max_num_iterations = 10;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(0.1);

    int kfsize = keyframelist.size();
    double t_array[kfsize][2];
    double r_array[kfsize][1];
    double t_array_origin[kfsize][2];
    double r_array_origin[kfsize][1];

    ceres::LocalParameterization* angle_local_parameterization =
            AngleLocalParameterization::Create();// 欧拉角的local参数化，防止超出±180

    // 用更新后的值作为初值

    Eigen::Vector3d pose0_before_opt;
    pose0_before_opt = keyframelist[earliest_loop_index]->GetPoseUpdate();
    for(int kfidx = earliest_loop_index, i = 0; kfidx < kfsize; kfidx++, i++)
    {
        Eigen::Vector3d tmp_pose = keyframelist[kfidx]->GetPoseUpdate();
        t_array[i][0] = tmp_pose(0);
        t_array[i][1] = tmp_pose(1);
        r_array[i][0] = tmp_pose(2);

        // 获取origin位姿
        tmp_pose = keyframelist[kfidx]->GetPose();
        t_array_origin[i][0] = tmp_pose(0);
        t_array_origin[i][1] = tmp_pose(1);
        r_array_origin[i][0] = tmp_pose(2);

        // 欧拉角表旋转进行优化
        problem.AddParameterBlock(r_array[i], 1, angle_local_parameterization);
        problem.AddParameterBlock(t_array[i], 2);

//        if(kfidx == earliest_loop_index)
//        {
//            problem.SetParameterBlockConstant(r_array[i]);
//            problem.SetParameterBlockConstant(t_array[i]);
//        }

        for(int j = 1; j < 5; j++)
        {
            if(i - j >= 0)
            {
                // origin's tij and rij
                double w_tij_x = t_array_origin[i][0] - t_array_origin[i-j][0];
                double w_tij_y = t_array_origin[i][1] - t_array_origin[i-j][1];
                double i_j_theta = r_array_origin[i-j][0] * M_PI / 180.0;
                double tij_x = cos(i_j_theta)*w_tij_x + sin(i_j_theta)*w_tij_y;
                double tij_y = -sin(i_j_theta)*w_tij_x + cos(i_j_theta)*w_tij_y;
                double rij = r_array_origin[i][0] - r_array_origin[i-j][0];

                ceres::CostFunction* cost_function = ThreeDOFError::Create(tij_x, tij_y, rij, 1.f, 0.1f);

                // 优化的参数包括第i帧的旋转/平移和第i-j帧的旋转/平移
                problem.AddResidualBlock(cost_function, nullptr, r_array[i-j], t_array[i-j], r_array[i], t_array[i]);
            }
        }

        const int loopid = keyframelist[kfidx]->loop_index;
        if(loopid >= 0)// 如果该帧有回环帧
        {
            Eigen::Vector3d pose_ij = keyframelist[kfidx]->loop_pose_ij;
            int connected_index = loopid - earliest_loop_index;
            ceres::CostFunction* cost_function = ThreeDOFError::Create(pose_ij.x(), pose_ij.y(), pose_ij.z(), 1.f, 0.1f);
            problem.AddResidualBlock(cost_function, loss_function, r_array[connected_index],
                                     t_array[connected_index], r_array[i], t_array[i]);

        }
    }
    ceres::Solve(options, &problem, &summary);
    Eigen::Vector3d pose0_after_opt = {t_array[0][0], t_array[0][1], r_array[0][0]};

    double d_r = pose0_before_opt.z() - pose0_after_opt.z();
    double d_theta = M_PI*d_r/180.0;
    Eigen::Matrix2d dR;
    dR << cos(d_theta), -sin(d_theta),
        sin(d_theta), cos(d_theta);
    Eigen::Vector2d dt = - dR*pose0_after_opt.head<2>() + pose0_before_opt.head<2>();

    // 优化后的关键帧赋值回去，首先是earliest_loop_index到cur_loop_opt_index更新
    for(int kfidx = earliest_loop_index + 1, i = 1; kfidx < kfsize; kfidx++, i++)
    {
        double r = r_array[i][0];
        Eigen::Vector2d t = {t_array[i][0], t_array[i][1]};
        double recursive_r = d_r + r;
        Eigen::Vector2d recursive_t = dR*t + dt;
        Eigen::Vector3d recursive_pose;
        recursive_pose << recursive_t.x(), recursive_t.y(), recursive_r;
        keyframelist[kfidx]->SetPoseUpdate(recursive_pose);
    }

    Eigen::Vector3d origin_pose, update_pose;
    origin_pose = keyframelist.back()->GetPose();
    update_pose = keyframelist.back()->GetPoseUpdate();
    pose_drift = Transform2D(update_pose, InverseTransform2D(origin_pose));

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

Eigen::Vector3d PoseGraph::DriftRemove(const Eigen::Vector3d& pose)
{
    return Transform2D(pose_drift, pose);
}

vector<shared_ptr<KeyFrame>> PoseGraph::GetKeyframelist()
{
    unique_lock<mutex> lock(m_keyframelist);
    return keyframelist;
}

void PoseGraph::SaveMap()
{
    string dbow3_data_path = save_path + "map.gz";
    db.save(dbow3_data_path);
    for(auto &kf : keyframelist)
    {
        int idx = kf->GetIndex();
        Eigen::Vector3d update_pose = kf->GetPoseUpdate();

        vector<cv::KeyPoint> keypoints = kf->GetKeypoints();
        vector<cv::Point2f> keypoints_norm = kf->GetKeypoints_norm();
        vector<float> keypoints_depth = kf->keypoints_depth;



        ofstream foutC(save_path + to_string(idx) + ".txt", ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(6);
        foutC << update_pose.x() << " "
              << update_pose.y() << " "
              << update_pose.z() << endl;
        foutC.precision(2);

        for(int i = 0; i < keypoints.size(); ++i)
        {
            foutC << keypoints[i].pt.x << " "
                  << keypoints[i].pt.y << " "
                  << keypoints_norm[i].x << " "
                  << keypoints_norm[i].y << " "
                  << keypoints_depth[i] << endl;
        }
        foutC.close();
        cv::Mat descriptors = kf->GetDescriptors();
        cv::imwrite(save_path + to_string(idx) + ".png", descriptors);
    }

    std::cout << "Successfully save map to " << save_path << std::endl;
    std::cout << "Totally have " << keyframelist.size() << " frames." << std::endl;
}

