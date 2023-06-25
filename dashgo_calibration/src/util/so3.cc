// SO3 FUNCTIONS

#include "so3.h"

// 取w^
Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w.z(), w.y(), w.z(), 0.0, -w.x(), -w.y(),  w.x(), 0.0;
    return W;
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() + W + 0.5*W*W;
    else
        return Eigen::Matrix3d::Identity() + W*sin(theta)/theta + W*W*(1.0-cos(theta))/theta2;
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    double costheta = 0.5*(R.trace()-1.0);
    if(costheta > +1.0) costheta = +1.0;
    if(costheta < -1.0) costheta = -1.0;
    const double theta = acos(costheta);// 先用迹的性质求转角theta，
    const Eigen::Vector3d w(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));// 先求转轴w，这里应该是2w，返回时*0.5
    if(theta < 1e-4)
        return 0.5*w;
    else
        return 0.5*theta*w/sin(theta);
}

Eigen::Matrix3d LeftJacobianSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() + 0.5*W + W*W/6.0;
    else
        return Eigen::Matrix3d::Identity() + W*(1.0-cos(theta))/theta2 + W*W*(theta-sin(theta))/(theta2*theta);
}

Eigen::Matrix3d InverseLeftJacobianSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() - 0.5*W + W*W/12.0;
    else
        return Eigen::Matrix3d::Identity() - 0.5*W + W*W*(1.0/theta2 - (1.0+cos(theta))/(2.0*theta*sin(theta)));
}


//Eigen::Matrix4d InverseSE3(const Eigen::Matrix4d &Ts)
//{
//    Eigen::Matrix4d res = Ts;
//    Eigen::Matrix3d inv_R = Ts.block(0, 0, 3, 3).transpose();
//    res.block(0, 0, 3, 3) = inv_R;
//    res.block(0, 3, 3, 1) = -inv_R*Ts.block(0, 3, 3, 1);
//    return res;
//}

Eigen::Matrix6d AdjointSE3(const Eigen::Isometry3d& Ts)
{
    Eigen::Matrix3d R = Ts.rotation();
    Eigen::Vector3d t = Ts.translation();
    Eigen::Matrix6d res;
    res.setZero();
    res.block(0, 0, 3, 3) = R;
    res.block(3, 3, 3, 3) = R;
    res.block(0, 3, 3, 3) = Skew(t)*R;
    return res;
}

Eigen::Isometry3d ExpSE3(const Eigen::Vector6d &t)
{
    Eigen::Vector3d fai = t.head(3);
    Eigen::Isometry3d Ts;
    Ts.setIdentity();
    Ts.linear() = ExpSO3(fai);
    Ts.translation() = LeftJacobianSO3(fai)*t.tail(3);
    return Ts;
}

Eigen::Vector6d LogSE3(const Eigen::Isometry3d &Ts)
{
    Eigen::Vector6d t;
    Eigen::Matrix3d R = Ts.rotation();
    t.head(3) = LogSO3(R);
    Eigen::Matrix3d InvJlogR = InverseLeftJacobianSO3(t.head(3));
    t.tail(3) = InvJlogR*Ts.translation();
    return t;
}

Eigen::Matrix6d LeftJacobianSE3(const Eigen::Vector6d &t)
{
    const Eigen::Vector3d r = t.head(3);
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 12.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (fai2 + 2*cos(fai) - 2)/(fai2*fai2);
        a3 = (2*fai - 3*sin(fai) + fai*cos(fai))/(2*fai2*fai2*fai);
    }

    const Eigen::Matrix3d Jr = LeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(t.tail(3));
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix6d res;
    res.setZero();
    res.block(0, 0, 3, 3) = Jr;
    res.block(3, 3, 3, 3) = Jr;
    res.block(0, 3, 3, 3) = Qrp;
    return res;
}

Eigen::Matrix6d InverseLeftJacobianSE3(const Eigen::Vector6d &t)
{
    const Eigen::Vector3d r = t.head(3);
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 12.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (fai2 + 2*cos(fai) - 2)/(fai2*fai2);
        a3 = (2*fai - 3*sin(fai) + fai*cos(fai))/(2*fai2*fai2*fai);
    }

    const Eigen::Matrix3d InvJr = InverseLeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(t.tail(3));
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix6d res;
    res.setZero();
    res.block(0, 0, 3, 3) = InvJr;
    res.block(3, 3, 3, 3) = InvJr;
    res.block(0, 3, 3, 3) = -InvJr*Qrp*InvJr;
    return res;
}



Eigen::Matrix5d FaiSE23(const Eigen::Matrix5d& tmp_r, const double t)
{
    Eigen::Matrix5d res = tmp_r;
    res.block(0, 4, 3, 1) = tmp_r.block(0, 4, 3, 1) + tmp_r.block(0, 3, 3, 1)*t;
    return res;
}

Eigen::Matrix9d F_Mat_SE23(const double t)
{
    Eigen::Matrix9d F;
    F.setIdentity();
    F.block(6, 3, 3, 3) = t*Eigen::Matrix3d::Identity();
    return F;
}

Eigen::Matrix5d InverseSE23(const Eigen::Matrix5d& tmp_r)
{
    Eigen::Matrix5d res = tmp_r;
    Eigen::Matrix3d inv_R = tmp_r.block(0, 0, 3, 3).transpose();
    res.block(0, 0, 3, 3) = inv_R;
    res.block(0, 3, 3, 1) = -inv_R*tmp_r.block(0, 3, 3, 1);
    res.block(0, 4, 3, 1) = -inv_R*tmp_r.block(0, 4, 3, 1);
    return res;
}
Eigen::Matrix9d AdjointSE23(const Eigen::Matrix5d& r)
{
    Eigen::Matrix3d R = r.block(0, 0, 3, 3);
    Eigen::Vector3d v = r.block(0, 3, 3, 1);
    Eigen::Vector3d p = r.block(0, 4, 3, 1);
    Eigen::Matrix9d res;
    res.setZero();
    res.block(0, 0, 3, 3) = R;
    res.block(3, 3, 3, 3) = R;
    res.block(6, 6, 3, 3) = R;
    res.block(3, 0, 3, 3) = Skew(v)*R;
    res.block(6, 0, 3, 3) = Skew(p)*R;
    return res;
}
Eigen::Matrix5d ExpSE23(const Eigen::Vector9d& s)
{
    Eigen::Vector3d fai = s.head(3);
    Eigen::Matrix5d r;
    r.setIdentity();
    r.block(0, 0, 3, 3) = ExpSO3(fai);
    r.block(0, 3, 3, 1) = LeftJacobianSO3(fai)*s.segment(3, 3);
    r.block(0, 4, 3, 1) = LeftJacobianSO3(fai)*s.tail(3);
    return r;
}

Eigen::Vector9d LogSE23(const Eigen::Matrix5d& r)
{
    Eigen::Vector9d s;
    Eigen::Matrix3d R = r.block(0, 0, 3, 3);
    s.head(3) = LogSO3(R);
    Eigen::Matrix3d InvJlogR = InverseLeftJacobianSO3(s.head(3));
    s.segment(3, 3) = InvJlogR*r.block(0, 3, 3, 1);
    s.tail(3) = InvJlogR*r.block(0, 4, 3, 1);
    return s;
}

Eigen::Matrix9d LeftJacobianSE23(const Eigen::Vector9d& s)
{
    const Eigen::Vector3d r = s.head(3);
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 12.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (fai2 + 2*cos(fai) - 2)/(fai2*fai2);
        a3 = (2*fai - 3*sin(fai) + fai*cos(fai))/(2*fai2*fai2*fai);
    }

    const Eigen::Matrix3d Jr = LeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_v = Skew(s.segment(3, 3));
    const Eigen::Matrix3d Skew_rvr = Skew_r*Skew_v*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(s.tail(3));
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrv = 0.5*Skew_v + a1*(Skew_r*Skew_v + Skew_v*Skew_r + Skew_rvr)
            + a2*(Skew_r2*Skew_v + Skew_v*Skew_r2 - 3*Skew_rvr)
            + a3*(Skew_rvr*Skew_r + Skew_r*Skew_rvr);
    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix9d res;
    res.setZero();
    res.block(0, 0, 3, 3) = Jr;
    res.block(3, 3, 3, 3) = Jr;
    res.block(6, 6, 3, 3) = Jr;
    res.block(3, 0, 3, 3) = Qrv;
    res.block(6, 0, 3, 3) = Qrp;
    return res;
}

Eigen::Matrix9d InverseLeftJacobianSE23(const Eigen::Vector9d& s)
{
    const Eigen::Vector3d r = s.head(3);
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 12.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (fai2 + 2*cos(fai) - 2)/(fai2*fai2);
        a3 = (2*fai - 3*sin(fai) + fai*cos(fai))/(2*fai2*fai2*fai);
    }

    const Eigen::Matrix3d InvJr = InverseLeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_v = Skew(s.segment(3, 3));
    const Eigen::Matrix3d Skew_rvr = Skew_r*Skew_v*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(s.tail(3));
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrv = 0.5*Skew_v + a1*(Skew_r*Skew_v + Skew_v*Skew_r + Skew_rvr)
            + a2*(Skew_r2*Skew_v + Skew_v*Skew_r2 - 3*Skew_rvr)
            + a3*(Skew_rvr*Skew_r + Skew_r*Skew_rvr);
    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix9d res;
    res.setZero();
    res.block(0, 0, 3, 3) = InvJr;
    res.block(3, 3, 3, 3) = InvJr;
    res.block(6, 6, 3, 3) = InvJr;
    res.block(3, 0, 3, 3) = -InvJr*Qrv*InvJr;
    res.block(6, 0, 3, 3) = -InvJr*Qrp*InvJr;
    return res;
}


// 已知两个方向向量，求旋转矩阵
// v2.normalized = R*v1.normalized;
Eigen::Matrix3d R_v2v(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
{
    const Eigen::Vector3d v1n = v1.normalized();
    const Eigen::Vector3d v2n = v2.normalized();
    const Eigen::Vector3d axis = (v1n.cross(v2n)).normalized();// 转轴
    const double theta = std::acos(v1n.dot(v2n));// 转角
    return ( ExpSO3(axis*theta) );
}

Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

Eigen::Matrix3d ypr2R(const Eigen::Vector3d &ypr)
{

    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Eigen::Matrix3d Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Eigen::Matrix3d Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}


double NormalizeAngle(const double& angle_degrees)
{
    if(angle_degrees > 180.0)
        return angle_degrees - 360.0;
    else if (angle_degrees < - 180.0)
        return angle_degrees + 360.0;
    else
        return angle_degrees;
}


Eigen::Quaterniond deltaQ(const Eigen::Map<const Eigen::Vector3d> &theta)
{
    Eigen::Quaterniond dq;
    Eigen::Vector3d half_theta = 0.5*theta;
    dq.w() = 1.0;
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

Eigen::Matrix4d Qleft(const Eigen::Quaterniond &q)
{
    Eigen::Matrix4d ans;
    ans(0, 0) = q.w();
    ans.block<3, 1>(1, 0) = q.vec();
    ans.block<1, 3>(0, 1) = q.vec().transpose();
    ans.block<3, 3>(1, 1) = q.w()*Eigen::Matrix3d::Identity() + Skew(q.vec());
    return ans;
}

Eigen::Matrix4d Qright(const Eigen::Quaterniond &q)
{
    Eigen::Matrix4d ans;
    ans(0, 0) = q.w();
    ans.block<3, 1>(1, 0) = q.vec();
    ans.block<1, 3>(0, 1) = q.vec().transpose();
    ans.block<3, 3>(1, 1) = q.w() * Eigen::Matrix3d::Identity() - Skew(q.vec());
    return ans;
}

//三点确定一个平面 a(x-x0)+b(y-y0)+c(z-z0)=0  --> ax + by + cz + d = 0   d = -(ax0 + by0 + cz0)
//平面通过点（x0,y0,z0）以及垂直于平面的法线（a,b,c）来得到    d = -(a, b, c)·(x0, y0, z0)
// 这里是求平面的参数(a, b, c, d)
Eigen::Vector4d ppp2pi(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3)
{
    Eigen::Vector4d pi;
    pi << (p1 - p3).cross(p2 - p3), - p3.dot(p1.cross(p2)); // d = - x3.dot( (x1-x3).cross( x2-x3 ) ) = - x3.dot( x1.cross( x2 ) )
    return pi;
}

// 两平面相交得到直线的plucker坐标
/*
 *
 *      | v^   n |       其中  n = Op1 × Op2
 *  L = |        |            v = Op2 - Op1
 *      |-nT   0 |
 *
 *
 * */
Eigen::Vector6d pipi2pluck(Eigen::Vector4d pi1, Eigen::Vector4d pi2)
{
    Eigen::Vector6d pluck;
    Eigen::Matrix4d dp = pi1*pi2.transpose() - pi2*pi1.transpose();
    pluck << dp(0, 3), dp(1, 3), dp(2, 3), dp(2, 1), dp(0, 2), dp(1, 0);//plucker坐标，先n后v
    return pluck;
}

// 获取当前帧下，光心到直线的垂直点的坐标
Eigen::Vector3d plucker_origin(Eigen::Vector3d n, Eigen::Vector3d v)
{
    return v.cross(n) / v.dot(v);
}

// 线特征，pluck到Orth坐标的转换（这里orth前三项是rpy）
Eigen::Vector4d pluck2orth(Eigen::Vector6d pluck)
{
    Eigen::Vector4d orth;
    Eigen::Vector3d n = pluck.head(3);
    Eigen::Vector3d v = pluck.tail(3);

    Eigen::Vector3d u1 = n/n.norm();
    Eigen::Vector3d u2 = v/v.norm();
    Eigen::Vector3d u3 = u1.cross(u2);

    orth[0] = atan2(u2(2), u3(2));
    orth[1] = asin(-u1(2));
    orth[2] = atan2(u1(1), u1(0));

    Eigen::Vector2d w(n.norm(), v.norm());
    w = w/w.norm();
    orth[3] = asin(w(1));

    return orth;
}
// 线特征，Orth到pluck坐标的转换（这里orth前三项是rpy）
Eigen::Vector6d orth2pluck(Eigen::Vector4d orth)
{
    Eigen::Vector6d pluck;
    Eigen::Vector3d theta = orth.head(3);
    double s1 = sin(theta[0]);
    double c1 = cos(theta[0]);
    double s2 = sin(theta[1]);
    double c2 = cos(theta[1]);
    double s3 = sin(theta[2]);
    double c3 = cos(theta[2]);

    Eigen::Matrix3d R;// 前三项欧拉角变换成旋转矩阵
    R << c2*c3,   s1*s2*c3 - c1*s3,   c1*s2*c3 + s1*s3,
         c2*s3,   s1*s2*s3 + c1*c3,   c1*s2*s3 - s1*c3,
          - s2,              s1*c2,              c1*c2;

    Eigen::Vector3d u1 = R.col(0);// 这里是 n / ||n||
    Eigen::Vector3d u2 = R.col(1);// 这里是 v / ||v||

    double phi = orth[3];
    double w1 = cos(phi);// 这里是 ||n|| / sqrt( ||n||^2 + ||v||^2 )
    double w2 = sin(phi);// 这里是 ||v|| / sqrt( ||n||^2 + ||v||^2 )

    pluck.head(3) = w1*u1;// 这里是 n / sqrt( ||n||^2 + ||v||^2 )
    pluck.tail(3) = w2*u2;// 这里是 v / sqrt( ||n||^2 + ||v||^2 )

    return pluck;// pluck坐标和尺度无关？
}

// pluck坐标系的变换
Eigen::Vector6d pluck_transform(Eigen::Vector6d pluck_c, Eigen::Matrix3d Rwc, Eigen::Vector3d twc)
{
    Eigen::Vector3d nc = pluck_c.head(3);
    Eigen::Vector3d vc = pluck_c.tail(3);

    Eigen::Vector6d pluck_w;
    pluck_w.head(3) = Rwc*nc + Skew(twc)*Rwc*vc;
    pluck_w.tail(3) = Rwc*vc;

    return pluck_w;
}

