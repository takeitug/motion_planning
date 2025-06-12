#include "motion_planning/manipulability_gradient.h"

namespace manipulability {

//manipulability
double calcmanipulability(const Eigen::MatrixXd &jacobian){
    Eigen::MatrixXd jj_t=jacobian*jacobian.transpose();
    return sqrt(jj_t.determinant());
}

Eigen::Matrix<double, 6, 7> Jq1(const Eigen::VectorXd &joint_position){
    double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    double J11=- dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) - dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - dse*cos(th1)*sin(th2);
    double J21=dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - dse*sin(th1)*sin(th2);
    double J31=0;
    double J41=0;
    double J51=0;
    double J61=0;
    double J12=-sin(th1)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) + dse*cos(th2));
    double J22=cos(th1)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) + dse*cos(th2));
    double J32=0;
    double J42=-cos(th1);
    double J52=-sin(th1);
    double J62=0;
    double J13=cos(th1)*sin(th2)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) + dse*cos(th2)) - cos(th2)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + dse*cos(th1)*sin(th2));
    double J23=cos(th2)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - dse*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) + dse*cos(th2));
    double J33=0;
    double J43=-sin(th1)*sin(th2);
    double J53=cos(th1)*sin(th2);
    double J63=0;
    double J14=(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) + sin(th2)*sin(th3)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)));
    double J24=- (cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - sin(th2)*sin(th3)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)));
    double J34=0;
    double J44=cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3);
    double J54=cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3);
    double J64=0;
    double J15=(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - (dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4));
    double J25=(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) - (sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))));
    double J35=0;
    double J45=sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2);
    double J55=sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2);
    double J65=0;
    double J16=dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J26=- dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J36=0;
    double J46=sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3));
    double J56=sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J66=0;
    double J17=0;
    double J27=0;
    double J37=0;
    double J47=cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J57=cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J67=0;

    Eigen::Matrix<double, 6, 1> J1;
    J1 << J11, J21, J31, J41, J51, J61;
    Eigen::Matrix<double, 6, 1> J2;
    J2 << J12, J22, J32, J42, J52, J62;
    Eigen::Matrix<double, 6, 1> J3;
    J3 << J13, J23, J33, J43, J53, J63;
    Eigen::Matrix<double, 6, 1> J4;
    J4 << J14, J24, J34, J44, J54, J64;
    Eigen::Matrix<double, 6, 1> J5;
    J5 << J15, J25, J35, J45, J55, J65;
    Eigen::Matrix<double, 6, 1> J6;
    J6 << J16, J26, J36, J46, J56, J66;
    Eigen::Matrix<double, 6, 1> J7;
    J7 << J17, J27, J37, J47, J57, J67;

    Eigen::Matrix<double, 6, 7> Jacobi;
    Jacobi << J1, J2, J3, J4, J5, J6, J7;

    return Jacobi;
}

Eigen::Matrix<double, 6, 7> Jq2(const Eigen::VectorXd &joint_position){
    double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    double J11=- dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) - dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))) - dse*cos(th2)*sin(th1);
    double J21=dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + dse*cos(th1)*cos(th2);
    double J31=0;
    double J41=0;
    double J51=0;
    double J61=0;
    double J12=-cos(th1)*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4)) + dse*sin(th2));
    double J22=-sin(th1)*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4)) + dse*sin(th2));
    double J32=- cos(th1)*(dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + dse*cos(th1)*cos(th2)) - sin(th1)*(dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))) + dse*cos(th2)*sin(th1));
    double J42=0;
    double J52=0;
    double J62=0;
    double J13=cos(th2)*sin(th1)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) + dse*cos(th2)) - cos(th2)*(dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))) + dse*cos(th2)*sin(th1)) - sin(th2)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - dse*sin(th1)*sin(th2)) - sin(th1)*sin(th2)*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4)) + dse*sin(th2));
    double J23=cos(th2)*(dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + dse*cos(th1)*cos(th2)) - sin(th2)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + dse*cos(th1)*sin(th2)) - cos(th1)*cos(th2)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) + dse*cos(th2)) + cos(th1)*sin(th2)*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4)) + dse*sin(th2));
    double J33=cos(th1)*sin(th2)*(dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))) + dse*cos(th2)*sin(th1)) - sin(th1)*sin(th2)*(dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + dse*cos(th1)*cos(th2)) - cos(th1)*cos(th2)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - dse*sin(th1)*sin(th2)) - cos(th2)*sin(th1)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + dse*cos(th1)*sin(th2));
    double J43=cos(th1)*cos(th2);
    double J53=cos(th2)*sin(th1);
    double J63=-sin(th2);
    double J14=(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + sin(th2)*sin(th3)*(dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)))) - cos(th2)*sin(th3)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) - sin(th1)*sin(th2)*sin(th3)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))));
    double J24=(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) - cos(th2)*sin(th3)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) - sin(th2)*sin(th3)*(dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4))) + cos(th1)*sin(th2)*sin(th3)*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))));
    double J34=(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4))) + (dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))))*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)) + sin(th1)*sin(th2)*sin(th3)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + cos(th1)*sin(th2)*sin(th3)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)));
    double J44=-cos(th1)*sin(th2)*sin(th3);
    double J54=-sin(th1)*sin(th2)*sin(th3);
    double J64=-cos(th2)*sin(th3);
    double J15=(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) - (cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + (cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - (dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4));
    double J25=(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) + dew*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) - (dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)))*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4)) - (cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) + (cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))*(dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)));
    double J35=(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(dwf*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dew*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4))) - (cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4))*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + (sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(dew*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)))) - (cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)));
    double J45=cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4);
    double J55=cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4);
    double J65=cos(th2)*cos(th3)*sin(th4) - cos(th4)*sin(th2);
    double J16=dwf*(sin(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) + cos(th2)*cos(th5)*sin(th3))*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))*(sin(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) - cos(th5)*sin(th1)*sin(th2)*sin(th3)) + dwf*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4)))*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4)));
    double J26=dwf*(sin(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) + cos(th2)*cos(th5)*sin(th3))*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4))) - dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))*(sin(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) - cos(th1)*cos(th5)*sin(th2)*sin(th3));
    double J36=dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4))) + dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))))*(sin(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) - cos(th1)*cos(th5)*sin(th2)*sin(th3)) + dwf*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5))) + dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))))*(sin(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) - cos(th5)*sin(th1)*sin(th2)*sin(th3));
    double J46=cos(th1)*cos(th5)*sin(th2)*sin(th3) - sin(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2));
    double J56=cos(th5)*sin(th1)*sin(th2)*sin(th3) - sin(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2));
    double J66=sin(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) + cos(th2)*cos(th5)*sin(th3);
    double J17=0;
    double J27=0;
    double J37=0;
    double J47=cos(th6)*(cos(th1)*cos(th2)*cos(th4) + cos(th1)*cos(th3)*sin(th2)*sin(th4)) + sin(th6)*(cos(th5)*(cos(th1)*cos(th2)*sin(th4) - cos(th1)*cos(th3)*cos(th4)*sin(th2)) + cos(th1)*sin(th2)*sin(th3)*sin(th5));
    double J57=sin(th6)*(cos(th5)*(cos(th2)*sin(th1)*sin(th4) - cos(th3)*cos(th4)*sin(th1)*sin(th2)) + sin(th1)*sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4)*sin(th1) + cos(th3)*sin(th1)*sin(th2)*sin(th4));
    double J67=- sin(th6)*(cos(th5)*(sin(th2)*sin(th4) + cos(th2)*cos(th3)*cos(th4)) - cos(th2)*sin(th3)*sin(th5)) - cos(th6)*(cos(th4)*sin(th2) - cos(th2)*cos(th3)*sin(th4));

    Eigen::Matrix<double, 6, 1> J1;
    J1 << J11, J21, J31, J41, J51, J61;
    Eigen::Matrix<double, 6, 1> J2;
    J2 << J12, J22, J32, J42, J52, J62;
    Eigen::Matrix<double, 6, 1> J3;
    J3 << J13, J23, J33, J43, J53, J63;
    Eigen::Matrix<double, 6, 1> J4;
    J4 << J14, J24, J34, J44, J54, J64;
    Eigen::Matrix<double, 6, 1> J5;
    J5 << J15, J25, J35, J45, J55, J65;
    Eigen::Matrix<double, 6, 1> J6;
    J6 << J16, J26, J36, J46, J56, J66;
    Eigen::Matrix<double, 6, 1> J7;
    J7 << J17, J27, J37, J47, J57, J67;

    Eigen::Matrix<double, 6, 7> Jacobi;
    Jacobi << J1, J2, J3, J4, J5, J6, J7;

    return Jacobi;
}

Eigen::Matrix<double, 6, 7> Jq3(const Eigen::VectorXd &joint_position){
    double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    double J11=dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3));
    double J21=dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J31=0;
    double J41=0;
    double J51=0;
    double J61=0;
    double J12=cos(th1)*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4));
    double J22=sin(th1)*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4));
    double J32=sin(th1)*(dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - cos(th1)*(dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J42=0;
    double J52=0;
    double J62=0;
    double J13=cos(th2)*(dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + sin(th1)*sin(th2)*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4));
    double J23=cos(th2)*(dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) - cos(th1)*sin(th2)*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4));
    double J33=- cos(th1)*sin(th2)*(dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - sin(th1)*sin(th2)*(dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J43=0;
    double J53=0;
    double J63=0;
    double J14=(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - (cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4)) - sin(th2)*sin(th3)*(dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - cos(th3)*sin(th2)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)));
    double J24=(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - (cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4)) - sin(th2)*sin(th3)*(dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) - cos(th3)*sin(th2)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)));
    double J34=(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3))*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) - (dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)))*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + (dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)) - (dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J44=cos(th1)*cos(th2)*cos(th3) - sin(th1)*sin(th3);
    double J54=cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1);
    double J64=-cos(th3)*sin(th2);
    double J15=(dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) - (sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4)) - sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - sin(th2)*sin(th3)*sin(th4)*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)));
    double J25=(dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) - (sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(dwf*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dew*sin(th2)*sin(th3)*sin(th4)) - sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - sin(th2)*sin(th3)*sin(th4)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)));
    double J35=(dwf*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dew*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - (sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(dwf*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dew*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + sin(th4)*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)))*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)) - sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)));
    double J45=sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J55=-sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3));
    double J65=-sin(th2)*sin(th3)*sin(th4);
    double J16=dwf*(cos(th3)*cos(th5)*sin(th2) - cos(th4)*sin(th2)*sin(th3)*sin(th5))*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))*(cos(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + cos(th4)*sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J26=dwf*(cos(th3)*cos(th5)*sin(th2) - cos(th4)*sin(th2)*sin(th3)*sin(th5))*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) - dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4)) - dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) - dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))*(cos(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th4)*sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J36=dwf*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) - dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))))*(cos(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + cos(th4)*sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))))*(cos(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th4)*sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J46=cos(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th4)*sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J56=- cos(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3));
    double J66=cos(th3)*cos(th5)*sin(th2) - cos(th4)*sin(th2)*sin(th3)*sin(th5);
    double J17=0;
    double J27=0;
    double J37=0;
    double J47=sin(th6)*(sin(th5)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th4)*cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + cos(th6)*sin(th4)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J57=- sin(th6)*(sin(th5)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - cos(th6)*sin(th4)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3));
    double J67=sin(th6)*(cos(th3)*sin(th2)*sin(th5) + cos(th4)*cos(th5)*sin(th2)*sin(th3)) - cos(th6)*sin(th2)*sin(th3)*sin(th4);

    Eigen::Matrix<double, 6, 1> J1;
    J1 << J11, J21, J31, J41, J51, J61;
    Eigen::Matrix<double, 6, 1> J2;
    J2 << J12, J22, J32, J42, J52, J62;
    Eigen::Matrix<double, 6, 1> J3;
    J3 << J13, J23, J33, J43, J53, J63;
    Eigen::Matrix<double, 6, 1> J4;
    J4 << J14, J24, J34, J44, J54, J64;
    Eigen::Matrix<double, 6, 1> J5;
    J5 << J15, J25, J35, J45, J55, J65;
    Eigen::Matrix<double, 6, 1> J6;
    J6 << J16, J26, J36, J46, J56, J66;
    Eigen::Matrix<double, 6, 1> J7;
    J7 << J17, J27, J37, J47, J57, J67;

    Eigen::Matrix<double, 6, 7> Jacobi;
    Jacobi << J1, J2, J3, J4, J5, J6, J7;

    return Jacobi;
}

Eigen::Matrix<double, 6, 7> Jq4(const Eigen::VectorXd &joint_position){
    double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    double J11=dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4));
    double J21=dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4));
    double J31=0;
    double J41=0;
    double J51=0;
    double J61=0;
    double J12=-cos(th1)*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))));
    double J22=-sin(th1)*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))));
    double J32=sin(th1)*(dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4))) - cos(th1)*(dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)));
    double J42=0;
    double J52=0;
    double J62=0;
    double J13=cos(th2)*(dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4))) - sin(th1)*sin(th2)*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))));
    double J23=cos(th2)*(dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4))) + cos(th1)*sin(th2)*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))));
    double J33=- sin(th1)*sin(th2)*(dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4))) - cos(th1)*sin(th2)*(dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)));
    double J43=0;
    double J53=0;
    double J63=0;
    double J14=(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - sin(th2)*sin(th3)*(dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)));
    double J24=(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - sin(th2)*sin(th3)*(dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)));
    double J34=(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4))) - (dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)))*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J44=0;
    double J54=0;
    double J64=0;
    double J15=(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - (cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2))*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) - (cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) + (dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4));
    double J25=(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(dew*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + dwf*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - (cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4))*(dew*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)))) - (dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)))*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + (cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))*(dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)));
    double J35=(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(dwf*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dew*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4))) - (dwf*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dew*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)))*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + (cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4))*(dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dew*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) - (cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4))*(dwf*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) + dew*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)));
    double J45=cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4);
    double J55=- cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - sin(th1)*sin(th2)*sin(th4);
    double J65=cos(th3)*cos(th4)*sin(th2) - cos(th2)*sin(th4);
    double J16=dwf*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) - dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) + dwf*sin(th5)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) - dwf*sin(th5)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J26=dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) - dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) + dwf*sin(th5)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) - dwf*sin(th5)*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4));
    double J36=dwf*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))) - dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))) - dwf*sin(th5)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dwf*sin(th5)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J46=-sin(th5)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2));
    double J56=sin(th5)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2));
    double J66=-sin(th5)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4));
    double J17=0;
    double J27=0;
    double J37=0;
    double J47=cos(th6)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + cos(th5)*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2));
    double J57=- cos(th6)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2));
    double J67=cos(th5)*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) - cos(th6)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2));

    Eigen::Matrix<double, 6, 1> J1;
    J1 << J11, J21, J31, J41, J51, J61;
    Eigen::Matrix<double, 6, 1> J2;
    J2 << J12, J22, J32, J42, J52, J62;
    Eigen::Matrix<double, 6, 1> J3;
    J3 << J13, J23, J33, J43, J53, J63;
    Eigen::Matrix<double, 6, 1> J4;
    J4 << J14, J24, J34, J44, J54, J64;
    Eigen::Matrix<double, 6, 1> J5;
    J5 << J15, J25, J35, J45, J55, J65;
    Eigen::Matrix<double, 6, 1> J6;
    J6 << J16, J26, J36, J46, J56, J66;
    Eigen::Matrix<double, 6, 1> J7;
    J7 << J17, J27, J37, J47, J57, J67;

    Eigen::Matrix<double, 6, 7> Jacobi;
    Jacobi << J1, J2, J3, J4, J5, J6, J7;

    return Jacobi;
}

Eigen::Matrix<double, 6, 7> Jq5(const Eigen::VectorXd &joint_position){
    double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    double J11=dwf*sin(th6)*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J21=dwf*sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J31=0;
    double J41=0;
    double J51=0;
    double J61=0;
    double J12=-dwf*cos(th1)*sin(th6)*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3));
    double J22=-dwf*sin(th1)*sin(th6)*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3));
    double J32=dwf*sin(th1)*sin(th6)*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - dwf*cos(th1)*sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J42=0;
    double J52=0;
    double J62=0;
    double J13=dwf*cos(th2)*sin(th6)*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - dwf*sin(th1)*sin(th2)*sin(th6)*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3));
    double J23=dwf*cos(th2)*sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) + dwf*cos(th1)*sin(th2)*sin(th6)*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3));
    double J33=- dwf*cos(th1)*sin(th2)*sin(th6)*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - dwf*sin(th1)*sin(th2)*sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J43=0;
    double J53=0;
    double J63=0;
    double J14=dwf*sin(th6)*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)) - dwf*sin(th2)*sin(th3)*sin(th6)*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J24=dwf*sin(th6)*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)) - dwf*sin(th2)*sin(th3)*sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J34=dwf*sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)) - dwf*sin(th6)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J44=0;
    double J54=0;
    double J64=0;
    double J15=dwf*sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) + dwf*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3));
    double J25=dwf*sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)) + dwf*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3));
    double J35=dwf*sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))) - dwf*sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J45=0;
    double J55=0;
    double J65=0;
    double J16=- dwf*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5))*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J26=- dwf*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5))*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) - dwf*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) + cos(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J36=dwf*(cos(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))))*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))) - dwf*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) - sin(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J46=cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3));
    double J56=- cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3));
    double J66=- cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - sin(th2)*sin(th3)*sin(th5);
    double J17=0;
    double J27=0;
    double J37=0;
    double J47=sin(th6)*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J57=-sin(th6)*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J67=-sin(th6)*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3));

    Eigen::Matrix<double, 6, 1> J1;
    J1 << J11, J21, J31, J41, J51, J61;
    Eigen::Matrix<double, 6, 1> J2;
    J2 << J12, J22, J32, J42, J52, J62;
    Eigen::Matrix<double, 6, 1> J3;
    J3 << J13, J23, J33, J43, J53, J63;
    Eigen::Matrix<double, 6, 1> J4;
    J4 << J14, J24, J34, J44, J54, J64;
    Eigen::Matrix<double, 6, 1> J5;
    J5 << J15, J25, J35, J45, J55, J65;
    Eigen::Matrix<double, 6, 1> J6;
    J6 << J16, J26, J36, J46, J56, J66;
    Eigen::Matrix<double, 6, 1> J7;
    J7 << J17, J27, J37, J47, J57, J67;

    Eigen::Matrix<double, 6, 7> Jacobi;
    Jacobi << J1, J2, J3, J4, J5, J6, J7;

    return Jacobi;
}

Eigen::Matrix<double, 6, 7> Jq6(const Eigen::VectorXd &joint_position){
    double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    double J11=-dwf*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J21=-dwf*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))));
    double J31=0;
    double J41=0;
    double J51=0;
    double J61=0;
    double J12=dwf*cos(th1)*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J22=dwf*sin(th1)*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J32=dwf*cos(th1)*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) - dwf*sin(th1)*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J42=0;
    double J52=0;
    double J62=0;
    double J13=dwf*sin(th1)*sin(th2)*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) - dwf*cos(th2)*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J23=- dwf*cos(th2)*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) - dwf*cos(th1)*sin(th2)*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J33=dwf*sin(th1)*sin(th2)*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) + dwf*cos(th1)*sin(th2)*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J43=0;
    double J53=0;
    double J63=0;
    double J14=dwf*sin(th2)*sin(th3)*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J24=dwf*sin(th2)*sin(th3)*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) - dwf*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J34=dwf*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))))*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3));
    double J44=0;
    double J54=0;
    double J64=0;
    double J15=- dwf*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) - dwf*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3))));
    double J25=- dwf*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4))) - dwf*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))))*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4));
    double J35=dwf*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2))*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2))*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))));
    double J45=0;
    double J55=0;
    double J65=0;
    double J16=dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J26=dwf*(sin(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) - cos(th5)*sin(th2)*sin(th3))*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))) - dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4)));
    double J36=dwf*(sin(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)))*(sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)))) - dwf*(sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3))))*(sin(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) - cos(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J46=0;
    double J56=0;
    double J66=0;
    double J17=0;
    double J27=0;
    double J37=0;
    double J47=- sin(th6)*(sin(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) + cos(th1)*cos(th4)*sin(th2)) - cos(th6)*(cos(th5)*(cos(th4)*(sin(th1)*sin(th3) - cos(th1)*cos(th2)*cos(th3)) - cos(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th3)*sin(th1) + cos(th1)*cos(th2)*sin(th3)));
    double J57=sin(th6)*(sin(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) - cos(th4)*sin(th1)*sin(th2)) + cos(th6)*(cos(th5)*(cos(th4)*(cos(th1)*sin(th3) + cos(th2)*cos(th3)*sin(th1)) + sin(th1)*sin(th2)*sin(th4)) + sin(th5)*(cos(th1)*cos(th3) - cos(th2)*sin(th1)*sin(th3)));
    double J67=cos(th6)*(cos(th5)*(cos(th2)*sin(th4) - cos(th3)*cos(th4)*sin(th2)) + sin(th2)*sin(th3)*sin(th5)) - sin(th6)*(cos(th2)*cos(th4) + cos(th3)*sin(th2)*sin(th4));

    Eigen::Matrix<double, 6, 1> J1;
    J1 << J11, J21, J31, J41, J51, J61;
    Eigen::Matrix<double, 6, 1> J2;
    J2 << J12, J22, J32, J42, J52, J62;
    Eigen::Matrix<double, 6, 1> J3;
    J3 << J13, J23, J33, J43, J53, J63;
    Eigen::Matrix<double, 6, 1> J4;
    J4 << J14, J24, J34, J44, J54, J64;
    Eigen::Matrix<double, 6, 1> J5;
    J5 << J15, J25, J35, J45, J55, J65;
    Eigen::Matrix<double, 6, 1> J6;
    J6 << J16, J26, J36, J46, J56, J66;
    Eigen::Matrix<double, 6, 1> J7;
    J7 << J17, J27, J37, J47, J57, J67;

    Eigen::Matrix<double, 6, 7> Jacobi;
    Jacobi << J1, J2, J3, J4, J5, J6, J7;

    return Jacobi;
}

Eigen::Matrix<double, 6, 7> Jq7(const Eigen::VectorXd &joint_position){
    double dbs = 0.34, dse = 0.4, dew = 0.4, dwf = 0.126;
    double th1 = joint_position[0], th2 = joint_position[1], th3 = joint_position[2], th4 = joint_position[3], th5 = joint_position[4], th6 = joint_position[5], th7 = joint_position[6];

    double J11=0;
    double J21=0;
    double J31=0;
    double J41=0;
    double J51=0;
    double J61=0;
    double J12=0;
    double J22=0;
    double J32=0;
    double J42=0;
    double J52=0;
    double J62=0;
    double J13=0;
    double J23=0;
    double J33=0;
    double J43=0;
    double J53=0;
    double J63=0;
    double J14=0;
    double J24=0;
    double J34=0;
    double J44=0;
    double J54=0;
    double J64=0;
    double J15=0;
    double J25=0;
    double J35=0;
    double J45=0;
    double J55=0;
    double J65=0;
    double J16=0;
    double J26=0;
    double J36=0;
    double J46=0;
    double J56=0;
    double J66=0;
    double J17=0;
    double J27=0;
    double J37=0;
    double J47=0;
    double J57=0;
    double J67=0;

    Eigen::Matrix<double, 6, 1> J1;
    J1 << J11, J21, J31, J41, J51, J61;
    Eigen::Matrix<double, 6, 1> J2;
    J2 << J12, J22, J32, J42, J52, J62;
    Eigen::Matrix<double, 6, 1> J3;
    J3 << J13, J23, J33, J43, J53, J63;
    Eigen::Matrix<double, 6, 1> J4;
    J4 << J14, J24, J34, J44, J54, J64;
    Eigen::Matrix<double, 6, 1> J5;
    J5 << J15, J25, J35, J45, J55, J65;
    Eigen::Matrix<double, 6, 1> J6;
    J6 << J16, J26, J36, J46, J56, J66;
    Eigen::Matrix<double, 6, 1> J7;
    J7 << J17, J27, J37, J47, J57, J67;

    Eigen::Matrix<double, 6, 7> Jacobi;
    Jacobi << J1, J2, J3, J4, J5, J6, J7;

    return Jacobi;
}

Eigen::VectorXd calctrace(const Eigen::MatrixXd &jq1, const Eigen::MatrixXd &jq2, const Eigen::MatrixXd &jq3, const Eigen::MatrixXd &jq4, const Eigen::MatrixXd &jq5, const Eigen::MatrixXd &jq6, const Eigen::MatrixXd &jq7, const Eigen::MatrixXd &j_inv){
    Eigen::MatrixXd Jq1_inv=jq1*j_inv;
    Eigen::MatrixXd Jq2_inv=jq2*j_inv;
    Eigen::MatrixXd Jq3_inv=jq3*j_inv;
    Eigen::MatrixXd Jq4_inv=jq4*j_inv;
    Eigen::MatrixXd Jq5_inv=jq5*j_inv;
    Eigen::MatrixXd Jq6_inv=jq6*j_inv;
    Eigen::MatrixXd Jq7_inv=jq7*j_inv;

    Eigen::VectorXd trace;
    trace<<Jq1_inv.trace(),Jq2_inv.trace(),Jq3_inv.trace(),Jq4_inv.trace(),Jq5_inv.trace(),Jq6_inv.trace(),Jq7_inv.trace();

    return trace;
}

Eigen::VecterXd gradient(const double &manip, const Eigen::VecterXd &trace_vec){
    return manip*trace_vec;
}


}