#ifndef MANIPULABILITY_INVERSE_H
#define MANIPULABILITY_INVERSE_H

#include "eigen3/Eigen/Dense"

namespace inversekinematics {

//jacobian calculation
Eigen::Matrix<double, 6, 7> calcJacobian(const Eigen::VectorXd &joint_position);

//pseudo-inverse calculation
Eigen::Matrix<double, 7, 6> calcJacobianInverse(const Eigen::MatrixXd &jacobian);

}  // namespace inversekinematics

#endif  // MANIPULABILITY_INVERSE_H
