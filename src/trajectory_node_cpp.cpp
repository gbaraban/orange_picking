#include "gcop/utils.h"
//#include "gcop/params.h"
#include "gcop/body3dcost.h"
#include "gcop/body3dwaypointcost.h"
#include "gcop/multicost.h"
#include "gcop/cylinder.h"
#include "gcop/groundplane.h"
#include "gcop/yawvelocityconstraint.h"
#include "gcop/constraintcost.h"
#include "gcop/yawcost.h"
#include "gcop/rpcost.h"
#include "gcop/direction_constraint.h"
#include "gcop/hrotor.h"
#include "gcop/ddp.h"
#include "gcop/so3.h"
#include <Eigen/Geometry>
#include <vector>
#include "ros/ros.h"
#include "nav_msgs/Path.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TransformStamped.h"
#include "geometry_msgs/Twist.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <string>

using namespace std;
using namespace Eigen;
using namespace gcop;

/**********************************
trajectory_node_cpp.cpp guide:
***********************************
This file defines a ROS node that runs GCOP trajectory generation on received goal point lists.
It contains the following functions:

void solver_process_goal(int N, double tf, int epochs, Body3dState x0,
     Body3dState goal1, Body3dState goal2, Body3dState goal3,
     Vector12d q, Vector12d qf, Vector4d r, vector<Body3dState> &xout, vector<Vector4d> &us)

void quat2Matrix(double x, double y, double z, double w, Matrix3d& R)

void quat2Matrix(tf2::Quaternion temp_quat, Matrix3d& R)

tf2::Quaternion matrix2Quat(Matrix3d& R)

void callback(const geometry_msgs::PoseArray::ConstPtr& msg)

TrajectoryNode(): lis(tfBuffer)

int main(int argc, char **argv)

To use this file, compile it with catkin and use rosrun or roslaunch to run "orange_picking trajectory_node_cpp"
When started, the main function will start a ROS node and call the TrajectoryNode constructor.
The constructor will check the ROS parameter server for the values of all pertinent params, then start listening
for goal messages.
In the callback for goal messages, the node will check TF for the current position/orientation of the system, then add the goals to TF as well.  If filter_alpha was specified as a parameter, it will perform a low-pass filter on 
the goals first.
After putting the goals into the world frame, the callback calls solver_process_goal, which is copied from 
hrotorpython.cc.  The values of the gains/weights for the cost function are hardcoded and should be double-checked
before use.  After using DDP to find an optimal trajectory, the result is published to a path topic.
********************************/

class TrajectoryNode
{

typedef Ddp<Body3dState, 12, 4> HrotorDdp;
typedef ConstraintCost<Body3dState, 12, 4, Dynamic, 3> DirectionConstraintCost;

public:

/*Runs DDP on the specified goals to create an optimal trejectory
Args:
N: The length (in points) of the desired trajectory
tf: The length (in time) of the desired trajectory
epochs: The number of DDP iterations to run
x0: The starting position
goal1, goal2, goal3: The desired waypoints.  They are assumed to be equally spaced in time.
q, r: The weights of the quadratic running cost
qf: The weights of the waypoint cost
xout, us: The vectors to populate with the resulting trajectory
*/
void solver_process_goal(int N, double tf, int epochs, Body3dState x0,
     Body3dState goal1, Body3dState goal2, Body3dState goal3,
     Vector12d q, Vector12d qf, Vector4d r, vector<Body3dState> &xout, vector<Vector4d> &us)
{
  //Parameters
  double h = tf/N;
  //System
  Hrotor sys;
  //Costs
  MultiCost<Body3dState, 12, 4> cost(sys, tf);
  //Quadratic Cost
  vector<Body3dState> goals(3);
  goals[0] = goal1;
  goals[1] = goal2;
  goals[2] = goal3;
  Matrix<double,12,12> Q;
  Matrix<double,12,12> Qf;
  Matrix<double,4,4> R;
  vector<double> time_list = {tf};//Use equal spacing
  Body3dCost<4> pathcost(sys, tf, goal3);
  Body3dWaypointCost waypointcost(sys, time_list, goals);
    for (int j = 0; j < 12; ++j)
    {
      pathcost.Q(j,j) = q[j];
      waypointcost.Q(j,j) = qf[j];
    }
    for (int j = 0; j < 4; ++j)
    {
      pathcost.R(j,j) = r[j];
    }
  cost.costs.push_back(&pathcost);
  cost.costs.push_back(&waypointcost);

  // Times
  vector<double> ts(N+1);
  for (int k = 0; k <= N; ++k)
    ts[k] = k*h;

  // States
  xout[0].Clear();
  xout[0] = x0;

  // initial controls (e.g. hover at one place)
  vector<Vector4d> uds(N);
  double third_N =((double) N)/goals.size();
  for (int i = 0; i < N; ++i) {
   // us[i].head(3).setZero();
   // us[i][3] = 9.81*sys.m;
    uds[i].head(3).setZero();
    uds[i][3] = 9.81*sys.m;
  }
  if (first_call) {
    for (int i = 0; i < N; ++i) {
      us[i].head(3).setZero();
      us[i][3] = 9.81*sys.m;
    }
  } 
  pathcost.SetReference(NULL,&uds);

  HrotorDdp ddp(sys, cost, ts, xout, us);
  ddp.mu = 1;
  ddp.debug = false;

  double lastV = -1;
  for (int ii = 0; ii < epochs; ++ii) {
    ddp.Iterate();
    ROS_INFO_STREAM("Iteration Num: " << ii << " DDP V: " << ddp.V << endl);
    if ((lastV != -1) && (lastV - ddp.V < 0.01)){
      break;
    }
    lastV = ddp.V;
  }
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
/*
void tf_update() {
  ros::Time begin_time = ros::Time::now();
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
  double tdiff = (curr_trans.header.stamp - last_time).toSec();
  if (tdiff < 1e-4){
    ROS_WARN("tf_update too soon");
    return;
  }
  //ctr++;
  Body3dState new_x0;
  new_x0.p << curr_trans.transform.translation.x,
          curr_trans.transform.translation.y,
          curr_trans.transform.translation.z;
  quat2Matrix(curr_trans.transform.rotation.x,
              curr_trans.transform.rotation.y,
              curr_trans.transform.rotation.z,
              curr_trans.transform.rotation.w,
              new_x0.R);
  last_time = ros::Time::now();//curr_trans.header.stamp;
  new_x0.v << (curr_trans.transform.translation.x - x0.p[0])/tdiff,
              (curr_trans.transform.translation.y - x0.p[1])/tdiff,
              (curr_trans.transform.translation.z - x0.p[2])/tdiff;
  filterX(new_x0);
  //ROS_INFO_STREAM("UPDATE Time Taken: " << (ros::Time::now()-begin_time).toSec() << std::endl);
 
}*/

void odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
  odom_update = true;
  x0.p << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z; 
  quat2Matrix(msg->pose.pose.orientation.x,
              msg->pose.pose.orientation.y,
              msg->pose.pose.orientation.z,
              msg->pose.pose.orientation.w,x0.R);
  x0.v << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z; 
  ROS_INFO_STREAM("ODOMETRY UPDATE");
  last_time = msg->header.stamp;
}
 
/*Handles messages of containing goal points
Args:
msg: The goal points message (should contain 3 points)
Assumes that the 3 goal points are associated with 1, 2, and 3 seconds in the future.
If the transform between world_name and matrice_name cannot be found within 5 seconds of message receipt, the
message is skipped and a warning is printed.
*/
void callback(const geometry_msgs::PoseArray::ConstPtr& msg)
{
//  ROS_INFO_STREAM("CALLBACK: " << ctr);
  if (!odom_update) {
    ROS_INFO_STREAM("Stale odometry value, retrying");
    return;
  }
  odom_update = false;
  ros::Time begin_time = ros::Time::now();
  //return;
  double tf = 3;
  int N = (int) 50*tf;
  int epochs = 4;
  //Body3dState x0;
  //Read x from TF
  //geometry_msgs::TransformStamped temp;
  //geometry_msgs::TransformStamped curr_trans;
  //geometry_msgs::TransformStamped past_trans;
  //ros::Time now = ros::Time::now();
  //ros::Time mid_tf;// = ros::Time::now();
  //ros::Time post_tf;// = ros::Time::now();
  /*try {
    curr_trans = tfBuffer.lookupTransform(world_name,matrice_name,ros::Time(0),ros::Duration(5.0));
    mid_tf = ros::Time::now();
  } catch (tf2::TransformException &ex) {
    ROS_WARN("Could not find the first transform");
    return;
  }
  try { 
    past_trans = tfBuffer.lookupTransform(world_name,matrice_name,now - tdiff,ros::Duration(1.0));
    post_tf = ros::Time::now();
  } catch (tf2::TransformException &ex) {
    ROS_WARN("Could not find the second transform");
    return;
  } 
  //Filter x
  x0.p << curr_trans.transform.translation.x,
          curr_trans.transform.translation.y,
          curr_trans.transform.translation.z;
  quat2Matrix(curr_trans.transform.rotation.x,
              curr_trans.transform.rotation.y,
              curr_trans.transform.rotation.z,
              curr_trans.transform.rotation.w,x0.R);
  x0.v << (curr_trans.transform.translation.x - past_trans.transform.translation.x)/tdiff.toSec(),
          (curr_trans.transform.translation.y - past_trans.transform.translation.y)/tdiff.toSec(),
          (curr_trans.transform.translation.z - past_trans.transform.translation.z)/tdiff.toSec();
  x0 = filterX(x0);*/
  //Check for all zero goals
  bool all_zeros = true;
  for (int ii = 0; ii < 3; ++ii) {
    if (msg->poses[ii].position.x != 0) {
      //std::cout << "x" << ii << " is non-zero" << std::endl;
      all_zeros = false;
      break;
    }
    if (msg->poses[ii].position.y != 0) {
      //std::cout << "y" << ii << " is non-zero" << std::endl;
      all_zeros = false;
      break;
    }
    if (msg->poses[ii].position.z != 0) {
      //std::cout << "z" << ii << " is non-zero" << std::endl;
      all_zeros = false;
      break;
    }
    if (msg->poses[ii].orientation.x != 0) {
      //std::cout << "Rx" << ii << " is non-zero" << std::endl;
      all_zeros = false;
      break;
    }
    if (msg->poses[ii].orientation.y != 0) {
      //std::cout << "Ry" << ii << " is non-zero" << std::endl;
      all_zeros = false;
      break;
    }
    if (msg->poses[ii].orientation.z != 0) {
      //std::cout << "Rz" << ii << " is non-zero" << std::endl;
      all_zeros = false;
      break;
    }
    if (msg->poses[ii].orientation.w != 1) {
      //std::cout << "Rw" << ii << " is non-one" << std::endl;
      all_zeros = false;
      break;
    }
  }
  /*if (all_zeros) {
    //std::cout << "Saw All Zeros!!" << std::endl;
  }*/
  Body3dState goal[3];
  for (int ii = 0; ii < 3; ++ii){
    Vector3d local_p;
    local_p << msg->poses[ii].position.x, msg->poses[ii].position.y, msg->poses[ii].position.z;
    Matrix3d local_R;
    quat2Matrix(msg->poses[ii].orientation.x,msg->poses[ii].orientation.y,msg->poses[ii].orientation.z,msg->poses[ii].orientation.w,local_R);
    Vector3d new_pos = x0.R*local_p + x0.p;
    Matrix3d new_rot = x0.R*local_R;
    if (first_call) {
      last_pos.push_back(new_pos);
      last_quat.push_back(matrix2Quat(new_rot));
    }
    if (!all_zeros) {
      goal[ii].p = (1 - filter_alpha)*last_pos[ii] + filter_alpha*new_pos;
      last_quat[ii] = last_quat[ii].slerp(matrix2Quat(new_rot),filter_alpha);
      quat2Matrix(last_quat[ii],goal[ii].R);
    } else {
      goal[ii].p = new_pos;
      last_quat[ii] = matrix2Quat(new_rot);
    }
    quat2Matrix(last_quat[ii],goal[ii].R);
    last_pos[ii] = goal[ii].p;
    goal[ii].v << 0, 0, 0;
    goal[ii].w << 0, 0, 0;
    geometry_msgs::TransformStamped temp;
    temp.header.stamp = last_time;
    temp.header.frame_id = world_name;
    temp.child_frame_id = goal_name + std::to_string(ii);
    temp.transform.translation.x = last_pos[ii][0];
    temp.transform.translation.y = last_pos[ii][1];
    temp.transform.translation.z = last_pos[ii][2];
    temp.transform.rotation.x = last_quat[ii].getX();
    temp.transform.rotation.y = last_quat[ii].getY();
    temp.transform.rotation.z = last_quat[ii].getZ();
    temp.transform.rotation.w = last_quat[ii].getW();
    br.sendTransform(temp);
  }
  Vector12d q;
  q << 0,0,0,0,0,0,15,15,15,10,10,10;
  Vector12d qf;
  qf << 10,10,10,10,10,10,0,0,0,0,0,0;
  Vector4d r;
  r << 0.01, 0.01, 0.01, 0.001;
  //vector<Body3dState> xs(N+1);
  //vector<Vector4d> us(N);
  ros::Time pre_ddp = ros::Time::now();
  solver_process_goal(N, tf, epochs, x0, goal[0], goal[1], goal[2],q, qf, r, xs, us);
  ros::Time post_ddp = ros::Time::now();
  //Make Path Message
  nav_msgs::Path pa;
  pa.poses.clear();
  pa.header.stamp = last_time;
  pa.header.frame_id = world_name;
  //Make Joint Message
  trajectory_msgs::JointTrajectory joint_msg;
  joint_msg.header.stamp = last_time;
  joint_msg.header.frame_id = world_name;
  joint_msg.points.clear();
  //Iterate through the states
  for (int ii = 0; ii < N+1; ++ii) {
    geometry_msgs::PoseStamped p;
    p.header.stamp = last_time;
    p.header.frame_id = "path"+std::to_string(ii);
    p.pose.position.x = xs[ii].p[0];
    p.pose.position.y = xs[ii].p[1];
    p.pose.position.z = xs[ii].p[2];
    tf2::Quaternion temp_quat = matrix2Quat(xs[ii].R);
    p.pose.orientation.x = temp_quat.getX();
    p.pose.orientation.y = temp_quat.getY();
    p.pose.orientation.z = temp_quat.getZ();
    p.pose.orientation.w = temp_quat.getW();
    pa.poses.push_back(p);
    trajectory_msgs::JointTrajectoryPoint joint_point;
    joint_point.positions.push_back(xs[ii].p[0]);
    joint_point.positions.push_back(xs[ii].p[1]);
    joint_point.positions.push_back(xs[ii].p[2]);
    double roll = SO3::Instance().roll(xs[ii].R);
    double pitch = SO3::Instance().pitch(xs[ii].R);
    double yaw = SO3::Instance().yaw(xs[ii].R);
    joint_point.positions.push_back(roll);
    joint_point.positions.push_back(pitch);
    joint_point.positions.push_back(yaw);
    joint_point.velocities.push_back(xs[ii].v[0]);
    joint_point.velocities.push_back(xs[ii].v[1]);
    joint_point.velocities.push_back(xs[ii].v[2]);
    double w0 = xs[ii].w[0];
    double w1 = xs[ii].w[1];
    double w2 = xs[ii].w[2];
    //NOTE: if cos(pitch) is small, this will break.
    double roll_rate = (w0*cos(yaw) + w1*sin(yaw))/cos(pitch);
    double pitch_rate = w1*cos(yaw) - w0*sin(yaw);
    double yaw_rate = (w2*cos(pitch) + w0*cos(yaw)*sin(pitch) + w1*sin(pitch)*sin(yaw))/cos(pitch);
    joint_point.velocities.push_back(roll_rate);
    joint_point.velocities.push_back(pitch_rate);
    joint_point.velocities.push_back(yaw_rate);
    if (ii < N) {
      double thrust = us[ii][3];
      double mass = 0.5; //Using the default
      Eigen:;Vector3d acc = xs[ii].R*Eigen::Vector3d(0,0,thrust/mass);      
      joint_point.accelerations.push_back(acc[0]);
      joint_point.accelerations.push_back(acc[1]);
      joint_point.accelerations.push_back(acc[2]-9.81);
    } else {
      joint_point.accelerations.push_back(0);
      joint_point.accelerations.push_back(0);
      joint_point.accelerations.push_back(0);
    }
    joint_msg.points.push_back(joint_point);
  }
  path_pub.publish(pa);
  joint_pub.publish(joint_msg);
  first_call = false;
  ROS_INFO_STREAM("x0 Position: " << last_pos[0] << std::endl);
  ROS_INFO_STREAM("x0 Velocity: " << x0velocity << std::endl);
  /*for (int ii = 0; ii < 3; ++ii) {
    ROS_INFO_STREAM("goal" << ii << " Position: " << last_pos[ii + 1] << std::endl);
    ROS_INFO_STREAM("goal" << ii << " Quaternion: " << last_quat[ii + 1] << std::endl);
  }*/
  ROS_INFO_STREAM("PreDDP Time Taken: " << (pre_ddp-last_time).toSec() << std::endl);
  ROS_INFO_STREAM("DDP Time Taken: " << (post_ddp - pre_ddp).toSec() << std::endl);
  ROS_INFO_STREAM("PostDDP Time Taken: " << (ros::Time::now()-post_ddp).toSec() << std::endl);
  ROS_INFO_STREAM("Total Time Taken: " << (ros::Time::now()-last_time).toSec() << std::endl);
}

/*Sets up of the node subscriber and publisher
Checks the following params:
path_topic: The topic to publish the path to.  Defaults to "path"
goal_topic: The topic to subscribe to for goals.  Defaults to "goal_points"
matrice_name: The quadcopter name in TF.  Defaults to "matrice"
world_name: The base transform name in TF.  Defaults to "world"
goal_name: The name to use for goal points in TF.  Defaults to "goal"
filter_alpha: Amount of filtering to use on the goal points.  Defaults to 1, which corresponds to no filtering.
*/
TrajectoryNode(): lis(tfBuffer)
{
  std::string path_topic, joint_topic, goal_topic;
  if (!(n.getParam("path_topic",path_topic))){
    path_topic = "path";
  }
  if (!(n.getParam("joint_path_topic",joint_topic))){
    joint_topic = "path_joints";
  }
  if (!(n.getParam("goal_topic",goal_topic))){
    goal_topic = "goal_points";
  }
  if (!(n.getParam("odom_topic",odom_topic))){
    odom_topic = "gcop_odom";
  }
  if (!(n.getParam("matrice_name",matrice_name))){
    matrice_name = "matrice";
  }
  if (!(n.getParam("world_name",world_name))){
    world_name = "world";
  }
  if (!(n.getParam("goal_name",goal_name))){
    goal_name = "goal";
  }
  if (!(n.getParam("filter_alpha",filter_alpha))){
    filter_alpha = 1;
  }
  ROS_INFO_STREAM("Publishing to " << path_topic << endl);
  ROS_INFO_STREAM("Publishing Joints to " << joint_topic << endl);
  ROS_INFO_STREAM("Subscribing to " << goal_topic << endl);
  ROS_INFO_STREAM("Subscribing to " << odom_topic << endl);
  ROS_INFO_STREAM("Matrice Name set to " << matrice_name << endl);
  ROS_INFO_STREAM("World Name set to " << world_name << endl);
  ROS_INFO_STREAM("Goal Name set to " << goal_name << endl);
  ROS_INFO_STREAM("Filter Alpha set to " << filter_alpha << endl);
  path_pub = n.advertise<nav_msgs::Path>(path_topic,1000);
  joint_pub = n.advertise<trajectory_msgs::JointTrajectory>(joint_topic,1000);
  sub = n.subscribe(goal_topic,1000,&TrajectoryNode::callback,this);
  odom_sub = n.subscribe(odom_topic,1000,&TrajectoryNode::odom_callback,this);
  int N = 150;
  x0.p << 0, 0, 0;
  x0.R << 1, 0, 0,
          0, 1, 0,
          0, 0, 1;
  x0.v << 0, 0, 0;
  xs = vector<Body3dState>(N+1);
  us = vector<Vector4d>(N);
  last_time = ros::Time::now();
  //xs.resize(N+1);
  //us.resize(N);
  first_call = true;
  odom_update = false;
/*  ros::Rate loop_rate(200);
  ctr = 0;
  ROS_INFO_STREAM("Loop Start: " << ctr);
  while (ros::ok())
  {
    tf_update();
    ros::spinOnce();
    loop_rate.sleep();
  }*/
}

tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener lis;
tf2_ros::TransformBroadcaster br;
ros::NodeHandle n;
ros::Publisher path_pub;
ros::Publisher joint_pub;
ros::Subscriber sub;
ros::Subscriber odom_sub;
std::string world_name;
std::string matrice_name;
std::string goal_name;
std::string odom_topic;
vector<Body3dState> xs;
vector<Vector4d> us;
double filter_alpha;
Body3dState x0;
vector<Vector3d> last_pos;
vector<tf2::Quaternion> last_quat;
Vector3d x0velocity;
bool first_call;
ros::Time last_time;
bool odom_update;
};
/*The main function
Builds an instance of TrajectoryNode and then spins.
*/
int main(int argc, char **argv)
{
  ros::init(argc,argv, "Trajectory_Node");
  TrajectoryNode tn;
  ros::spin();
  return 0;
}
