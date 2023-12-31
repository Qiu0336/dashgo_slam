
#ifndef SO3_H_
#define SO3_H_

#include <Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <Eigen/SVD>

namespace Eigen {
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 18, 18> Matrix18d;
} // namespace Eigen

Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);
inline Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{ return ExpSO3(Eigen::Vector3d(x, y, z)); }
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);
Eigen::Matrix3d LeftJacobianSO3(const Eigen::Vector3d &w);
inline Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &w) { return LeftJacobianSO3(-w); }
Eigen::Matrix3d InverseLeftJacobianSO3(const Eigen::Vector3d &w);
inline Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &w) { return InverseLeftJacobianSO3(-w); }



//Eigen::Matrix4d InverseSE3(const Eigen::Matrix4d &Ts);
Eigen::Matrix6d AdjointSE3(const Eigen::Isometry3d& Ts);
Eigen::Isometry3d ExpSE3(const Eigen::Vector6d &t);
Eigen::Vector6d LogSE3(const Eigen::Isometry3d &Ts);
Eigen::Matrix6d LeftJacobianSE3(const Eigen::Vector6d &t);
inline Eigen::Matrix6d RightJacobianSE3(const Eigen::Vector6d &t) { return LeftJacobianSE3(-t); }
Eigen::Matrix6d InverseLeftJacobianSE3(const Eigen::Vector6d &t);
inline Eigen::Matrix6d InverseRightJacobianSE3(const Eigen::Vector6d &t) { return InverseLeftJacobianSE3(-t); }




Eigen::Matrix5d FaiSE23(const Eigen::Matrix5d& tmp_r, const double t);
Eigen::Matrix9d F_Mat_SE23(const double t);
Eigen::Matrix5d InverseSE23(const Eigen::Matrix5d& tmp_r);
Eigen::Matrix9d AdjointSE23(const Eigen::Matrix5d& r);
Eigen::Matrix5d ExpSE23(const Eigen::Vector9d& s);
Eigen::Vector9d LogSE23(const Eigen::Matrix5d& r);
Eigen::Matrix9d LeftJacobianSE23(const Eigen::Vector9d& s);
inline Eigen::Matrix9d RightJacobianSE23(const Eigen::Vector9d& s) { return LeftJacobianSE23(-s); }
Eigen::Matrix9d InverseLeftJacobianSE23(const Eigen::Vector9d& s);
inline Eigen::Matrix9d InverseRightJacobianSE23(const Eigen::Vector9d& s) { return InverseLeftJacobianSE23(-s); }


Eigen::Quaterniond deltaQ(const Eigen::Map<const Eigen::Vector3d> &theta);
Eigen::Matrix4d Qleft(const Eigen::Quaterniond &q);
Eigen::Matrix4d Qright(const Eigen::Quaterniond &q);
Eigen::Matrix3d R_v2v(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R);
Eigen::Matrix3d ypr2R(const Eigen::Vector3d &ypr);
double NormalizeAngle(const double& angle_degrees);


Eigen::Vector4d ppp2pi(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3);
Eigen::Vector6d pipi2pluck(Eigen::Vector4d pi1, Eigen::Vector4d pi2);
Eigen::Vector3d plucker_origin(Eigen::Vector3d n, Eigen::Vector3d v);
Eigen::Vector4d pluck2orth(Eigen::Vector6d pluck);
Eigen::Vector6d orth2pluck(Eigen::Vector4d orth);
Eigen::Vector6d pluck_transform(Eigen::Vector6d pluck_c, Eigen::Matrix3d Rwc, Eigen::Vector3d twc);

#endif  // SO3_H_
