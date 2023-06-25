
#include "drawer.h"

void DrawNone()
{
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);// 启动深度测试
    glEnable(GL_BLEND);// 启动颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);// 颜色混合的方式

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 500),
                // 相机参数配置，高度，宽度，4个内参，最近/最远视距
                pangolin::ModelViewLookAt(2,0,2, 0,0,0, pangolin::AxisY)
                // 相机所在位置，相机所看点的位置，最后是相机轴方向
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
    // 显示视图在窗口中的范围（下上左右），最后一个参数为视窗长宽比

    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 清空颜色和深度缓存,刷新显示
        d_cam.Activate(s_cam);// 激活并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);// 画背景

        pangolin::FinishFrame();// 最终显示
    }
}


Drawer::Drawer()
{
    SetQuit(false);
}

void Drawer::SetParameter(cv::FileStorage fSettings)
{
    mBackgroundpatchsize = fSettings["view_PatchSize"];
    mBackgroundpatchcount = fSettings["view_PatchCount"];
    mPointSize = fSettings["view_PointSize"];
    mCameraSize = fSettings["view_CameraSize"];
    mCameraLineWidth = fSettings["view_CameraLineWidth"];
}

void Drawer::DrawBackground()
{
    int z = mBackgroundpatchcount;
    float w = mBackgroundpatchsize;
    float edge = z*w;
    float x;
    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    for(int i = 0; i <= z; i++)
    {
        x = w*i;
        glVertex3f(x, edge, 0);
        glVertex3f(x, -edge, 0);

        glVertex3f(-x, edge, 0);
        glVertex3f(-x, -edge, 0);

        glVertex3f(edge, x, 0);
        glVertex3f(-edge, x, 0);

        glVertex3f(edge, -x, 0);
        glVertex3f(-edge, -x, 0);
    }
    glEnd();
}


void Drawer::DrawCamera2D()
{
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    Eigen::Vector3d CurPose = GetCurPose();
    Eigen::Vector3d UpdateCurPose = mpPoseGraph->DriftRemove(CurPose);
    Eigen::Vector3d t;
    t << UpdateCurPose.x(), UpdateCurPose.y(), 0.5;
    Eigen::Matrix3d R = ypr2R(Eigen::Vector3d(UpdateCurPose.z() - 90, 0, 0));
    Twc.m[0] = R(0, 0);
    Twc.m[1] = R(1, 0);
    Twc.m[2] = R(2, 0);
    Twc.m[3] = 0.0;

    Twc.m[4] = R(0, 1);
    Twc.m[5] = R(1, 1);
    Twc.m[6] = R(2, 1);
    Twc.m[7] = 0.0;

    Twc.m[8] = R(0, 2);
    Twc.m[9] = R(1, 2);
    Twc.m[10] = R(2, 2);
    Twc.m[11] = 0.0;

    Twc.m[12] = t(0);
    Twc.m[13] = t(1);
    Twc.m[14] = t(2);
    Twc.m[15] = 1.0;

    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    glMultMatrixd(Twc.m);

    glLineWidth(mCameraLineWidth);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);

    glVertex3f(0, 0, 0);
    glVertex3f(w, h, -z);

    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, -z);

    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, h, -z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, h, -z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, h, -z);
    glVertex3f(w, h, -z);
    glEnd();

    glPopMatrix();
}


void Drawer::DrawTrajectory2D()
{
    const auto kflist = mpPoseGraph->GetKeyframelist();
    if(kflist.size() == 0)
        return;
    Eigen::Vector3d CurPose = GetCurPose();
    Eigen::Vector3d UpdateCurPose = mpPoseGraph->DriftRemove(CurPose);
    const int ptsize = mPointSize;
    glPointSize(ptsize/2);//点的大小
    glColor3f(0.0 ,1.0, 0.0);//颜色
    glBegin(GL_LINES);

    double height = 0.5;

    int trajsize = kflist.size();
    Eigen::Vector3d pos1;
    Eigen::Vector3d pos2;
    pos1 = kflist[0]->pose_update;
    for(int i = 1; i < trajsize; i++)
    {
        pos2 = kflist[i]->pose_update;
        glVertex3f(pos1(0), pos1(1), height);
        glVertex3f(pos2(0), pos2(1), height);
        if(kflist[i]->loop_index >= 0)
        {
            glColor3f(0.0 ,0.0, 1.0);//颜色
            int lp = kflist[i]->loop_index;
            Eigen::Vector3d posloop = kflist[lp]->pose_update;
            glVertex3f(posloop(0), posloop(1), height);
            glVertex3f(pos2(0), pos2(1), height);

            glColor3f(0.0 ,1.0, 0.0);//颜色
        }
        pos1 = pos2;
    }

    glVertex3f(pos2(0), pos2(1), height);
    glVertex3f(UpdateCurPose.x(), UpdateCurPose.y(), height);

    glEnd();
}


void Drawer::DrawTrajectory2D_without_loop()
{
    const auto kflist = mpPoseGraph->GetKeyframelist();
    if(kflist.size() == 0)
        return;
    Eigen::Vector3d CurPose = GetCurPose();
    const int ptsize = mPointSize;
    glPointSize(ptsize/2);//点的大小
    glColor3f(1.0, 0.0, 1.0);//颜色
    glBegin(GL_LINES);

    double height = 0.5;

    int trajsize = kflist.size();
    Eigen::Vector3d pos1;
    Eigen::Vector3d pos2;
    pos1 = kflist[0]->pose;
    for(int i = 1; i < trajsize; i++)
    {
        pos2 = kflist[i]->pose;
        glVertex3f(pos1(0), pos1(1), height);
        glVertex3f(pos2(0), pos2(1), height);
        pos1 = pos2;
    }

    glVertex3f(pos2(0), pos2(1), height);
    glVertex3f(CurPose.x(), CurPose.y(), height);

    glEnd();
}

void Drawer::Run()
{
    pangolin::CreateWindowAndBind("Main", 640, 480);
//    pangolin::CreateWindowAndBind("Main", 1024,768);
    glEnable(GL_DEPTH_TEST);// 启动深度测试
    glEnable(GL_BLEND);// 启动颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);// 颜色混合的方式

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 500),
                // 相机参数配置，高度，宽度，4个内参，最近/最远视距
                pangolin::ModelViewLookAt(-2,0,2, 0,0,0, pangolin::AxisZ)
                // 相机所在位置，相机所看点的位置，最后是相机轴方向
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
    // 显示视图在窗口中的范围（下上左右），最后一个参数为视窗长宽比

    while(!pangolin::ShouldQuit() && !GetQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 清空颜色和深度缓存,刷新显示
        d_cam.Activate(s_cam);// 激活并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);// 画背景

        DrawBackground();
        DrawCamera2D();
        DrawTrajectory2D();

        DrawTrajectory2D_without_loop();

        pangolin::FinishFrame();// 最终显示
    }
}


void Drawer::SetCurPose(const Eigen::Vector3d& cur_pose_2d_)
{
  unique_lock<mutex> lock(m_cur_pose_2d);
  cur_pose_2d = cur_pose_2d_;
}

Eigen::Vector3d Drawer::GetCurPose()
{
  unique_lock<mutex> lock(m_cur_pose_2d);
  return cur_pose_2d;
}


void Drawer::SetQuit(bool x)
{
    unique_lock<mutex> lock(m_quit);
    quit_flag = x;
}

bool Drawer::GetQuit()
{
    unique_lock<mutex> lock(m_quit);
    return quit_flag;
}

