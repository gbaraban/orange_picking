#include "nav_msgs/Odometry.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "ros/ros.h"
#include "gcop/so3.h"
#include "gcop/body3dcost.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <string>
#include <Eigen/Geometry>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace gcop;

class OdometryNode
{

public:

void filterX(Body3dState new_x0) {
/*  if (first_call) {
    last_pos.push_back(x0.p);
    last_quat.push_back(matrix2Quat(x0.R));
    x0velocity = x0.v;
    return x0;
  }*/
  x0.p = x0.p + (new_x0.p - new_x0.p)*filter_alpha;
  tf2::Quaternion temp_quat = matrix2Quat(x0.R);
  temp_quat.slerp(matrix2Quat(new_x0.R),filter_alpha);
  quat2Matrix(temp_quat,x0.R);
  x0.v = x0.v + (new_x0.v - x0.v)*filter_alpha;
}

/*Constructs a matrix out of a quaternion, using 4 doubles as inputs
Args:
x,y,z,w: The quaternion input
R: The matrix output
*/
void quat2Matrix(double x, double y, double z, double w, Matrix3d& R) {
  tf2::Quaternion temp_quat(x,y,z,w);
  tf2::Matrix3x3 temp_mat(temp_quat);
  R << temp_mat[0][0], temp_mat[0][1], temp_mat[0][2],
          temp_mat[1][0], temp_mat[1][1], temp_mat[1][2],
          temp_mat[2][0], temp_mat[2][1], temp_mat[2][2]; 
}

/*Constructs a matrix out of a quaternion, using a quaternion object as input
Args:
temp_quat: The quaternion input
R: The matrix output
*/
void quat2Matrix(tf2::Quaternion temp_quat, Matrix3d& R) {
  tf2::Matrix3x3 temp_mat(temp_quat);
  R << temp_mat[0][0], temp_mat[0][1], temp_mat[0][2],
          temp_mat[1][0], temp_mat[1][1], temp_mat[1][2],
          temp_mat[2][0], temp_mat[2][1], temp_mat[2][2]; 
}

/*Constructs a quaternion out of a matrix
Args:
R: The matrix input
*/
tf2::Quaternion matrix2Quat(Matrix3d& R) {
  tf2::Matrix3x3 temp_mat;
  for (int jj = 0; jj < 3; ++jj) {
    for (int kk = 0; kk < 3; ++kk) {
      double temp_val = R(jj,kk);
      temp_mat[jj][kk] = temp_val;
    }
  }
  tf2::Quaternion temp_quat;
  temp_mat.getRotation(temp_quat);
  return temp_quat;
}

void tf_update() {
  ctr++;
  //ros::Time begin_time = ros::Time::now();
  geometry_msgs::TransformStamped curr_trans;
  try {
    curr_trans = tfBuffer.lookupTransform(world_name,matrice_name,ros::Time(0),ros::Duration(1.0));
  } catch (tf2::TransformException &ex) {
    ROS_WARN("Could not find the transform");
    return;
  }
//  try { 
//    past_trans = tfBuffer.lookupTransform(world_name,matrice_name,now - tdiff,ros::Duration(1.0));
//    post_tf = ros::Time::now();
//  } catch (tf2::TransformException &ex) {
//    ROS_WARN("Could not find the second transform");
//    return;
//  } 
  //Filter x
  ros::Time now = ros::Time::now();
  double tdiff = (now - last_time).toSec();
  if (tdiff < 1e-4){
    ROS_WARN_STREAM("tf_update too soon: " << ctr << " " << tdiff);
    return;
  }
  ctr = 0;
  Body3dState new_x0;
  new_x0.p << curr_trans.transform.translation.x,
          curr_trans.transform.translation.y,
          curr_trans.transform.translation.z;
  quat2Matrix(curr_trans.transform.rotation.x,
              curr_trans.transform.rotation.y,
              curr_trans.transform.rotation.z,
              curr_trans.transform.rotation.w,
              new_x0.R);
  last_time = now;
  new_x0.v << (curr_trans.transform.translation.x - x0.p[0])/tdiff,
              (curr_trans.transform.translation.y - x0.p[1])/tdiff,
              (curr_trans.transform.translation.z - x0.p[2])/tdiff;
  filterX(new_x0);
  nav_msgs::Odometry temp;
  temp.header.stamp = now;
  temp.pose.pose.position.x = x0.p[0];
  temp.pose.pose.position.y = x0.p[1];
  temp.pose.pose.position.z = x0.p[2];
  tf2::Quaternion temp_quat = matrix2Quat(x0.R);
  temp.pose.pose.orientation.x = temp_quat.getX();
  temp.pose.pose.orientation.y = temp_quat.getY(); 
  temp.pose.pose.orientation.z = temp_quat.getZ(); 
  temp.pose.pose.orientation.w = temp_quat.getW(); 
  temp.twist.twist.linear.x = x0.v[0];
  temp.twist.twist.linear.y = x0.v[1];
  temp.twist.twist.linear.z = x0.v[2];
  odom_pub.publish(temp);
  ROS_INFO_STREAM("Sending Odometry: " << tdiff);
}

OdometryNode(): lis(tfBuffer)
{
  if (!(n.getParam("matrice_name",matrice_name))){
    matrice_name = "matrice";
  }
  if (!(n.getParam("world_name",world_name))){
    world_name = "world";
  }
  if (!(n.getParam("filter_alpha",filter_alpha))){
    filter_alpha = 1;
  }
  if (!(n.getParam("odom_topic",odom_topic))){
    odom_topic = "gcop_odom";
  }
  ctr = 0;
  odom_pub = n.advertise<nav_msgs::Odometry>(odom_topic,1000);
  ros::Rate loop_rate(100);
  while (ros::ok())
  {
    tf_update();
    ros::spinOnce();
    loop_rate.sleep();
  }
}


ros::NodeHandle n;
tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener lis;
Body3dState x0;
ros::Time last_time;
ros::Publisher odom_pub;
std::string world_name;
std::string matrice_name;
std::string odom_topic;
double filter_alpha;
double ctr;
bool first_time;
};

int main(int argc, char **argv)
{
  ros::init(argc,argv, "Odometry_Node");
  OdometryNode on;
  return 0;
}
  
