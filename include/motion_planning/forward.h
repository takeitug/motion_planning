#ifndef MANIPULABILITY_FORWARD_H
#define MANIPULABILITY_FORWARD_H

#include "eigen3/Eigen/Dense"

namespace forwardkinematics {

//jacobian calculation
Eigen::Matrix<double, 4, 4> calcfk(const Eigen::VectorXd &joint_position, double dwf);

}  // namespace forwardkinematics

#endif  // MANIPULABILITY_FORWARD_H
