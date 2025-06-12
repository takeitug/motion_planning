#ifndef MANIPULABILITY_GRADIENT_H
#define MANIPULABILITY_GRADIENT_H

#include "eigen3/Eigen/Dense"

namespace manipulability {

//manipulability
double calcmanipulability(const Eigen::MatrixXd &jacobian);

Eigen::Matrix<double, 6, 7> Jq1(const Eigen::VectorXd &joint_position);
Eigen::Matrix<double, 6, 7> Jq2(const Eigen::VectorXd &joint_position);
Eigen::Matrix<double, 6, 7> Jq3(const Eigen::VectorXd &joint_position);
Eigen::Matrix<double, 6, 7> Jq4(const Eigen::VectorXd &joint_position);
Eigen::Matrix<double, 6, 7> Jq5(const Eigen::VectorXd &joint_position);
Eigen::Matrix<double, 6, 7> Jq6(const Eigen::VectorXd &joint_position);
Eigen::Matrix<double, 6, 7> Jq7(const Eigen::VectorXd &joint_position);

Eigen::VectorXd calctrace(const Eigen::MatrixXd &jq1, const Eigen::MatrixXd &jq2, const Eigen::MatrixXd &jq3, const Eigen::MatrixXd &jq4, const Eigen::MatrixXd &jq5, const Eigen::MatrixXd &jq6, const Eigen::MatrixXd &jq7, const Eigen::MatrixXd &j_inv);

Eigen::VecterXd gradient(const double &manip, const Eigen::VecterXd &trace_vec);

}  // namespace manipulability

#endif  // MANIPULABILITY_GRADIENT_H
