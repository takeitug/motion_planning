#include "motion_planning/forward.h"

namespace forwardkinematics {

Eigen::Matrix<double, 4, 4> calcfk(const Eigen::VectorXd &joint_position, double dwf=0.126){
    // double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double dbs = 0.34, dse = 0.4, dew = 0.4;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    Eigen::Matrix<double, 4, 4> T01;
    T01 << cos(th1), 0, -sin(th1), 0,
        sin(th1), 0, cos(th1), 0,
        0, -1, 0, dbs,
        0, 0, 0, 1;

    Eigen::Matrix<double, 4, 4> T12;
    T12 << cos(th2), 0, sin(th2), 0,
        sin(th2), 0, -cos(th2), 0,
        0, 1, 0, 0,
        0, 0, 0, 1;

    Eigen::Matrix<double, 4, 4> T23;
    T23 << cos(th3), 0, sin(th3), 0,
        sin(th3), 0, -cos(th3), 0,
        0, 1, 0, dse,
        0, 0, 0, 1;

    Eigen::Matrix<double, 4, 4> T34;
    T34 << cos(th4), 0, -sin(th4), 0,
        sin(th4), 0, cos(th4), 0,
        0, -1, 0, 0,
        0, 0, 0, 1;

    Eigen::Matrix<double, 4, 4> T45;
    T45 << cos(th5), 0, -sin(th5), 0,
        sin(th5), 0, cos(th5), 0,
        0, -1, 0, dew,
        0, 0, 0, 1;

    Eigen::Matrix<double, 4, 4> T56;
    T56 << cos(th6), 0, sin(th6), 0,
        sin(th6), 0, -cos(th6), 0,
        0, 1, 0, 0,
        0, 0, 0, 1;

    Eigen::Matrix<double, 4, 4> T67;
    T67 << cos(th7), -sin(th7), 0, 0,
        sin(th7), cos(th7), 0, 0,
        0, 0, 1, dwf,
        0, 0, 0, 1;

    Eigen::Matrix<double, 4, 4> T07 = T01 * T12 * T23 * T34 * T45 * T56 * T67;

    return T07;
}

}  // namespace forwardkinematics
