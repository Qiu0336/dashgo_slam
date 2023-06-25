#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "util/so3.h"
#include "camera/camera.h"

// 旋转矩阵的重新参数化
class RotationMatrixParameterization : public ceres::LocalParameterization
{
    public:
    virtual ~RotationMatrixParameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Matrix3d> R(x);
        Eigen::Matrix3d delta_R = ExpSO3(delta[0], delta[1], delta[2]);
        Eigen::Map<Eigen::Matrix3d> result(x_plus_delta);
        result = R*delta_R;
        return true;
    }

// 新参数R相对于老参数的雅克比,这里是R对δα,δβ,δγ的一阶导数，用一阶泰勒展开（δα,δβ,δγ=0处）
// R = REXP(δα,δβ,δγ) = R + R*(δα,δβ,δγ)^;
// 小技巧：J * (δα,δβ,δγ).Transpose  =  R*(δα,δβ,δγ)^ 的按第一列，第二列，第三列展开
    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
//        Eigen::Map<const Eigen::Matrix3d> R(x);
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.topRows(3).setIdentity();
        return true;
    }
    int GlobalSize() const override { return 9; }// 参数的实际维度
    int LocalSize() const override { return 3; }// 正切空间上的维度，这里R实则3维
};


// SO3的本地参数化，设置参数更新方式
class SO3Parameterization : public ceres::LocalParameterization
{
    public:
    virtual ~SO3Parameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Vector3d> s(x);
        Eigen::Map<const Eigen::Vector3d> ds(delta);
        Eigen::Map<Eigen::Vector3d> update_s(x_plus_delta);
        update_s = LogSO3(ExpSO3(s)*ExpSO3(ds));
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }
    int GlobalSize() const override { return 3; }// 参数的实际维度
    int LocalSize() const override { return 3; }// 正切空间上的维度，这里R实则3维
};


// SE3的本地参数化，设置参数更新方式
class SE3Parameterization : public ceres::LocalParameterization
{
    public:
    virtual ~SE3Parameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Vector6d> s(x);
        Eigen::Map<const Eigen::Vector6d> ds(delta);
        Eigen::Map<Eigen::Vector6d> update_s(x_plus_delta);
        update_s = LogSE3(ExpSE3(s)*ExpSE3(ds));
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }
    int GlobalSize() const override { return 6; }// 参数的实际维度
    int LocalSize() const override { return 6; }// 正切空间上的维度，这里R实则3维
};

class BundlePnP: public ceres::SizedCostFunction<2, 9, 3>
{
    public:
    BundlePnP(Eigen::Vector3d _p3d, Eigen::Vector2d _p2d) : p3d(_p3d), p2d(_p2d)
    {
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix3d> Riw(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tiw(parameters[1]);

        const Eigen::Vector3d pi = Riw*p3d + tiw;
        const double dep = pi(2);
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = pi.hnormalized() - p2d;

        if(jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1.0/dep,       0, - pi(0)/(dep*dep),
                            0, 1.0/dep, - pi(1)/(dep*dep);

            if(jacobians[0])// 对ri求导
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - Riw*Skew(p3d);
                J.leftCols(3) = reduce*jaco;
                J.rightCols(6).setZero();
            }
            if(jacobians[1])// 对ti求导
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = reduce;
            }
        }
        return true;
    }
    Eigen::Vector3d p3d;
    Eigen::Vector2d p2d;
};


class FixedPose: public ceres::SizedCostFunction<2, 3>
{
    public:
    FixedPose(Eigen::Matrix3d& _Rwc, Eigen::Vector3d& _twc, Eigen::Vector2d& _puv_i) : Rwc(_Rwc), twc(_twc), puv_i(_puv_i)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> mappoint(parameters[0]);

        const Eigen::Vector3d pi = Rwc.transpose()*(mappoint - twc);
        const double dep = pi(2);
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = pi.hnormalized() - puv_i;

        if(jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1.0/dep,       0, - pi(0)/(dep*dep),
                            0, 1.0/dep, - pi(1)/(dep*dep);

            if(jacobians[0])// 对ti求导
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = reduce*Rwc.transpose();
            }
        }
        return true;
    }
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d twc;
    Eigen::Vector2d puv_i;
};

class RelaxedPose: public ceres::SizedCostFunction<2, 9, 3, 3>
{
    public:
    RelaxedPose(Eigen::Vector2d& _puv_i, Eigen::Matrix3d _Rji = Eigen::Matrix3d::Identity(),
                Eigen::Vector3d _tji = Eigen::Vector3d::Zero()) :
        puv_i(_puv_i), Rji(_Rji), tji(_tji)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix3d> Rcw(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tcw(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> mappoint(parameters[2]);

        const Eigen::Vector3d pi = Rji*(Rcw*mappoint + tcw) + tji;
        const double dep = pi(2);
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = pi.hnormalized() - puv_i;

        if(jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1.0/dep,       0, - pi(0)/(dep*dep),
                            0, 1.0/dep, - pi(1)/(dep*dep);

            if (jacobians[0])// 对Ri
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - Rji*Rcw*Skew(mappoint);
                J.leftCols(3) = reduce*jaco;
                J.rightCols(6).setZero();
            }
            if (jacobians[1])// 对ti
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = reduce*Rji;
            }
            if (jacobians[2])// 对Pi
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[2]);
                J = reduce*Rji*Rcw;
            }
        }
        return true;
    }
    Eigen::Vector2d puv_i;
    Eigen::Matrix3d Rji;
    Eigen::Vector3d tji;
};


// Traditional
class LoopTradition: public ceres::SizedCostFunction<6, 9, 3, 9, 3>
{
    public:
    LoopTradition(Eigen::Isometry3d &_Tij, double _w_t = 1.0f, double _w_R = 1.0f) : Tij(_Tij)
    {
        Rij = Tij.rotation();
        tij = Tij.translation();
        w_t = _w_t;
        w_R = _w_R;
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix3d> Rwi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> twi(parameters[1]);
        Eigen::Map<const Eigen::Matrix3d> Rwj(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> twj(parameters[3]);


        Eigen::Map<Eigen::Vector6d> residual(residuals);
        Eigen::Vector3d Log_er = LogSO3(Rij.transpose()*Rwi.transpose()*Rwj);
        residual.head(3) = Log_er*w_R;
        residual.tail(3) = (Rwi.transpose()*(twj - twi) - tij)*w_t;

        if(jacobians)
        {
            const Eigen::Matrix3d JrSO3_rij= InverseRightJacobianSO3(Log_er);
            if(jacobians[0])// 对ri求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 9, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                J.topLeftCorner(3, 3) = - w_R*JrSO3_rij*Rwj.transpose()*Rwi;
                J.bottomLeftCorner(3, 3) = Skew(Rwi.transpose()*(twj - twi))*w_t;
            }
            if(jacobians[1])// 对ti求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[1]);
                J.topRows(3).setZero();
                J.bottomRows(3) = -Rwi.transpose()*w_t;
            }
            if(jacobians[2])// 对rj求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 9, Eigen::RowMajor>> J(jacobians[2]);
                J.setZero();
                J.topLeftCorner(3, 3) = w_R*JrSO3_rij;
            }
            if(jacobians[3])// 对tj求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[3]);
                J.topRows(3).setZero();
                J.bottomRows(3) = Rwi.transpose()*w_t;
            }
        }
        return true;
    }
    Eigen::Isometry3d Tij;
    Eigen::Matrix3d Rij;
    Eigen::Vector3d tij;
    double w_t, w_R;
};



// SO3--6dof
class LoopSO3: public ceres::SizedCostFunction<6, 3, 3, 3, 3>
{
    public:
    LoopSO3(Eigen::Isometry3d &_Tij) : Tij(_Tij)
    {
        Rij = Tij.rotation();
        tij = Tij.translation();
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> ri(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> ti(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> rj(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> tj(parameters[3]);

        Eigen::Matrix3d Riw = ExpSO3(ri).transpose();
        Eigen::Matrix3d Rwj = ExpSO3(rj);


        Eigen::Map<Eigen::Vector6d> residual(residuals);
        Eigen::Vector3d Log_er = LogSO3(Rij.transpose()*Riw*Rwj);
        residual.head(3) = Log_er;
        residual.tail(3) = Riw*(tj - ti) - tij;

        if(jacobians)
        {
            const Eigen::Matrix3d JrSO3_rij= InverseRightJacobianSO3(Log_er);
            if(jacobians[0])// 对ri求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.topRows(3) = - JrSO3_rij*(Riw*Rwj).inverse();
                J.bottomRows(3) = Skew(Riw*(tj - ti));
            }
            if(jacobians[1])// 对ti求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[1]);
                J.topRows(3).setZero();
                J.bottomRows(3) = -Riw;
            }
            if(jacobians[2])// 对rj求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[2]);
                J.topRows(3) = JrSO3_rij;
                J.bottomRows(3).setZero();
            }
            if(jacobians[3])// 对tj求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[3]);
                J.topRows(3).setZero();
                J.bottomRows(3) = Riw;
            }
        }
        return true;
    }
    Eigen::Isometry3d Tij;
    Eigen::Matrix3d Rij;
    Eigen::Vector3d tij;
};



// SE3--6dof
class LoopSE3: public ceres::SizedCostFunction<6, 6, 6>
{
    public:
    LoopSE3(Eigen::Isometry3d &_Tij) : Tij(_Tij)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector6d> ti(parameters[0]);
        Eigen::Map<const Eigen::Vector6d> tj(parameters[1]);

//        const Eigen::Isometry3d Twi = ExpSE3(ti);
//        const Eigen::Isometry3d Twj = ExpSE3(tj);
//        const Eigen::Isometry3d vTji = Twj.inverse()*Twi;
        const Eigen::Isometry3d vTji = ExpSE3(tj).inverse()*ExpSE3(ti);

        Eigen::Map<Eigen::Vector6d> residual(residuals);
        Eigen::Vector6d Log_er = LogSE3(Tij*vTji);
        residual = Log_er;

        if(jacobians)
        {
            const Eigen::Matrix6d JrSE3_rij= InverseRightJacobianSE3(Log_er);
            if(jacobians[0])// 对ξi求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[0]);
                J = JrSE3_rij;
            }
            if(jacobians[1])// 对ξj求导
            {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[1]);
                J = - JrSE3_rij*AdjointSE3(vTji.inverse());
            }
        }
        return true;
    }
    Eigen::Isometry3d Tij;
};



// 以下是VINS的4dof优化
template <typename T>
T NormalizeAngle(const T& angle_degrees)
{
    if (angle_degrees > T(180.0))
        return angle_degrees - T(360.0);
    else if (angle_degrees < T(-180.0))
        return angle_degrees + T(360.0);
    else
        return angle_degrees;
};

class AngleLocalParameterization
{
    public:

    template <typename T>
    bool operator()(const T* theta_radians, const T* delta_theta_radians, T* theta_radians_plus_delta) const
    {
        *theta_radians_plus_delta = NormalizeAngle(*theta_radians + *delta_theta_radians);
        return true;
    }
    static ceres::LocalParameterization* Create()
    {
        return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>);
    }
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

    T y = yaw / T(180.0) * T(M_PI);
    T p = pitch / T(180.0) * T(M_PI);
    T r = roll / T(180.0) * T(M_PI);

    R[0] = cos(y) * cos(p);
    R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
    R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
    R[3] = sin(y) * cos(p);
    R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
    R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
    R[6] = -sin(p);
    R[7] = cos(p) * sin(r);
    R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
    inv_R[0] = R[0];
    inv_R[1] = R[3];
    inv_R[2] = R[6];
    inv_R[3] = R[1];
    inv_R[4] = R[4];
    inv_R[5] = R[7];
    inv_R[6] = R[2];
    inv_R[7] = R[5];
    inv_R[8] = R[8];
};

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
    r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
    r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
    r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

// 四自由度优化函数，优化时保证帧与帧间的相对位姿不变，相对yaw不变，绝对位姿可以变
// 传入的参数t_x(t_x), t_y(t_y), t_z(t_z)为tij是提前算好的相对位姿tij，relative_yaw是相对yaw，
// pitch_i和roll_i是固定的pitch和roll轴
// 优化的参数包括绝对位置ti，tj和两帧的yaw
struct FourDOFError
{
    FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i, double _wt, double _wr)
                  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i), weight_t(_wt) , weight_r(_wr){}

    template <typename T>
    bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        // euler to rotation
        T w_R_i[9];
        YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
        // rotation transpose
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        // rotation matrix rotate point
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);// Riw*(twj - twi) = tij（i系下）

        residuals[0] = (t_i_ij[0] - T(t_x))*weight_t;
        residuals[1] = (t_i_ij[1] - T(t_y))*weight_t;
        residuals[2] = (t_i_ij[2] - T(t_z))*weight_t;
        residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw)) * weight_r;

        return true;
    }

    static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                       const double relative_yaw, const double pitch_i, const double roll_i, const double _wt, const double _wr)
    {
      return (new ceres::AutoDiffCostFunction<
              FourDOFError, 4, 1, 3, 1, 3>(
                new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i, _wt, _wr)));
    }

    double t_x, t_y, t_z;
    double relative_yaw, pitch_i, roll_i;
    double weight_t, weight_r;

};


//struct ThreeDOFError
//{
//    ThreeDOFError(double t_x, double t_y, double relative_yaw, double _wt, double _wr)
//                  :t_x(t_x), t_y(t_y), relative_yaw(relative_yaw), weight_t(_wt) , weight_r(_wr){}

//    bool operator()(const double* const yaw_i, const double* ti, const double* yaw_j, const double* tj, double* residuals) const
//    {

//        double t_w_ij[2];// tij in w frame
//        t_w_ij[0] = tj[0] - ti[0];
//        t_w_ij[1] = tj[1] - ti[1];
//        double y_theta = M_PI* yaw_i[0] / 180.0;
//        double t_i_ij[2];// tij in i frame
//        t_i_ij[0] = cos(y_theta)*t_w_ij[0] + sin(y_theta)*t_w_ij[1];
//        t_i_ij[1] = -sin(y_theta)*t_w_ij[0] + cos(y_theta)*t_w_ij[1];


//        residuals[0] = (t_i_ij[0] - double(t_x))*weight_t;
//        residuals[1] = (t_i_ij[1] - double(t_y))*weight_t;
//        residuals[2] = NormalizeAngle(yaw_j[0] - yaw_i[0] - double(relative_yaw)) * weight_r;

//        return true;
//    }

//    static ceres::CostFunction* Create(const double t_x, const double t_y,
//                                       const double relative_yaw, const double _wt, const double _wr)
//    {
//      return (new ceres::AutoDiffCostFunction<
//              ThreeDOFError, 3, 1, 2, 1, 2>(
//                new ThreeDOFError(t_x, t_y, relative_yaw, _wt, _wr)));
//    }

//    double t_x, t_y;
//    double relative_yaw;
//    double weight_t, weight_r;
//};


struct ThreeDOFError
{
    ThreeDOFError(double t_x, double t_y, double relative_yaw, double _wt, double _wr)
                  :t_x(t_x), t_y(t_y), relative_yaw(relative_yaw), weight_t(_wt) , weight_r(_wr){}

    template <typename T>
    bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
    {
        T t_w_ij[2];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];


        T y_theta = yaw_i[0] / T(180.0) * T(M_PI);
        T t_i_ij[2];// tij in i frame
        t_i_ij[0] = cos(y_theta)*t_w_ij[0] + sin(y_theta)*t_w_ij[1];
        t_i_ij[1] = -sin(y_theta)*t_w_ij[0] + cos(y_theta)*t_w_ij[1];


        residuals[0] = (t_i_ij[0] - T(t_x))*weight_t;
        residuals[1] = (t_i_ij[1] - T(t_y))*weight_t;
        residuals[2] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw)) * weight_r;
        return true;
    }

    static ceres::CostFunction* Create(const double t_x, const double t_y,
                                       const double relative_yaw, const double _wt, const double _wr)
    {
      return (new ceres::AutoDiffCostFunction<
              ThreeDOFError, 3, 1, 2, 1, 2>(
                new ThreeDOFError(t_x, t_y, relative_yaw, _wt, _wr)));
    }

    double t_x, t_y;
    double relative_yaw;
    double weight_t, weight_r;

};
