#include "nav_msgs/Odometry.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "ros/ros.h"
#include "gcop/so3.h"
#include "gcop/body3dcost.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/TransformStamped.h"
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
  x0.p = x0.p + (new_x0.p - x0.p)*filter_alpha;
  tf2::Quaternion x0_quat = matrix2Quat(x0.R);
  tf2::Quaternion new_x0_quat = matrix2Quat(new_x0.R);
  x0_quat = x0_quat.slerp(new_x0_quat,filter_alpha);
  quat2Matrix(x0_quat,x0.R);
  //x0.R = new_x0.R;
  x0.v = x0.v + (new_x0.v - x0.v)*filter_alpha;
  x0.w = x0.w + (new_x0.w - x0.w)*filter_alpha;
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

void odom_update(const geometry_msgs::TransformStamped::ConstPtr& msg) {
//void tf_update() {
  ctr++;
  //ros::Time begin_time = ros::Time::now();
  geometry_msgs::TransformStamped curr_trans;
  curr_trans = (*msg);
/*  try {
    curr_trans = tfBuffer.lookupTransform(world_name,matrice_name,ros::Time(0),ros::Duration(1.0));
  } catch (tf2::TransformException &ex) {
    ROS_WARN("Could not find the transform");
    return;
  }*/
//  try { 
//    past_trans = tfBuffer.lookupTransform(world_name,matrice_name,now - tdiff,ros::Duration(1.0));
//    post_tf = ros::Time::now();
//  } catch (tf2::TransformException &ex) {
//    ROS_WARN("Could not find the second transform");
//    return;
//  } 
  //Filter x
  ros::Time now = curr_trans.header.stamp;//ros::Time::now();
  double tdiff = (now - last_time).toSec();
  if (tdiff < 1e-4){
    ROS_WARN_STREAM("tf_update too soon: " << ctr << " " << tdiff);
    return;
  }
  ctr = 0;
  Body3dState new_x0;
  //Populate new_x0.p and R
  new_x0.p << curr_trans.transform.translation.x,
          curr_trans.transform.translation.y,
          curr_trans.transform.translation.z;
  quat2Matrix(curr_trans.transform.rotation.x,
              curr_trans.transform.rotation.y,
              curr_trans.transform.rotation.z,
              curr_trans.transform.rotation.w,
              new_x0.R);
  /*Eigen::Matrix3d R_offset;
  R_offset << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  new_x0.R = new_x0.R*R_offset;*/
  //Filter new_x0.p and R
  new_x0.p = x0.p + (new_x0.p - x0.p)*filter_alpha;
  tf2::Quaternion x0_quat = matrix2Quat(x0.R);
  tf2::Quaternion new_x0_quat = matrix2Quat(new_x0.R);
  new_x0_quat = x0_quat.slerp(new_x0_quat,filter_alpha);
  quat2Matrix(new_x0_quat,new_x0.R);
  //Calculate v
  double dx = (curr_trans.transform.translation.x - x0.p[0]);
  if ((dx < 1e-5) && (dx > -1e-5)) {dx = 0;}
  double dy = (curr_trans.transform.translation.y - x0.p[1]);
  if ((dy < 1e-5) && (dy > -1e-5)) {dy = 0;}
  double dz = (curr_trans.transform.translation.z - x0.p[2]);
  if ((dz < 1e-5) && (dz > -1e-5)) {dz = 0;}
  new_x0.v << dx/tdiff,dy/tdiff,dz/tdiff;
  //Calculate w
  /*Eigen::Matrix3d temp_R;
  double theta = 0.7*tdiff;
  temp_R = x0.R*R_offset;
  SO3::Instance().log(new_x0.w, x0.R.transpose()*temp_R);*/
  SO3::Instance().log(new_x0.w, x0.R.transpose()*new_x0.R);
  new_x0.w = new_x0.R.transpose()*new_x0.w;
  for (int ii = 0; ii < 3; ++ii) {
    if ((new_x0.w[ii] < 1e-5) && (new_x0.w[ii] > -1e-5)) { new_x0.w[ii] = 0;}
    else {new_x0.w[ii] = new_x0.w[ii]/tdiff;}
  }
  //Filter v and w
  new_x0.v = x0.v + (new_x0.v - x0.v)*filter_alpha;
  new_x0.w = x0.w + (new_x0.w - x0.w)*filter_alpha;
  //Set x0 for next time
  x0.p = new_x0.p;
  x0.R = new_x0.R;
  x0.v = new_x0.v;
  x0.w = new_x0.w;
  //filterX(new_x0);
  //Send odometry
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
  temp.twist.twist.angular.x = x0.w[0];
  temp.twist.twist.angular.y = x0.w[1];
  temp.twist.twist.angular.z = x0.w[2];
  odom_pub.publish(temp);
  //Publish TF
  geometry_msgs::TransformStamped temp_tf;
  temp_tf.header.stamp = now;
  temp_tf.header.frame_id = world_name;
  temp_tf.child_frame_id = "x0";
  temp_tf.transform.translation.x = x0.p[0];
  temp_tf.transform.translation.y = x0.p[1];
  temp_tf.transform.translation.z = x0.p[2];
  temp_tf.transform.rotation.x = temp_quat.getX();
  temp_tf.transform.rotation.y = temp_quat.getY();
  temp_tf.transform.rotation.z = temp_quat.getZ();
  temp_tf.transform.rotation.w = temp_quat.getW();
  br.sendTransform(temp_tf);
  last_time = now;
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
  if (!(n.getParam("vrpn_topic",vrpn_topic))){
    vrpn_topic = "vrpn_client/matrice/pose";
  }
  ctr = 0;
  x0.p << 0,0,0;
  x0.R << 1,0,0,0,1,0,0,0,1;
  x0.v << 0,0,0;
  odom_pub = n.advertise<nav_msgs::Odometry>(odom_topic,1);
  vrpn_sub = n.subscribe(vrpn_topic,1,&OdometryNode::odom_update,this);
  /*ros::Rate loop_rate(50);
  while (ros::ok())
  {
    tf_update();
    ros::spinOnce();
    loop_rate.sleep();
  }*/
}


ros::NodeHandle n;
tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener lis;
tf2_ros::TransformBroadcaster br;
Body3dState x0;
ros::Time last_time;
ros::Publisher odom_pub;
ros::Subscriber vrpn_sub;
std::string world_name;
std::string matrice_name;
std::string odom_topic;
std::string vrpn_topic;
double filter_alpha;
double ctr;
bool first_time;
};

int main(int argc, char **argv)
{
  ros::init(argc,argv, "Odometry_Node");
  OdometryNode on;
  ros::spin();
  return 0;
}
  
