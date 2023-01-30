#include <Eigen/Geometry>

Eigen::Matrix3d kabsch(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out);
Eigen::Vector3d RottoDof(Eigen::Matrix3d R);
void TestDOF(Eigen::Matrix3d R);