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
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TransformStamped.h"
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
  xout.resize(N+1);
  xout[0].Clear();
  xout[0] = x0;

  // initial controls (e.g. hover at one place)
  us.resize(N);
  vector<Vector4d> uds(N);
  double third_N =((double) N)/goals.size();
  for (int i = 0; i < N; ++i) {
    us[i].head(3).setZero();
    us[i][3] = 9.81*sys.m;
    uds[i].head(3).setZero();
    uds[i][3] = 9.81*sys.m;
  }
  pathcost.SetReference(NULL,&uds);

  HrotorDdp ddp(sys, cost, ts, xout, us);
  ddp.mu = 1;
  ddp.debug = false;

  for (int ii = 0; ii < epochs; ++ii) {
    ddp.Iterate();
    cout << " Iteration Num: " << ii << " DDP V: " << ddp.V << endl;
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
 
/*Handles messages of containing goal points
Args:
msg: The goal points message (should contain 3 points)
Assumes that the 3 goal points are associated with 1, 2, and 3 seconds in the future.
If the transform between world_name and matrice_name cannot be found within 5 seconds of message receipt, the
message is skipped and a warning is printed.
*/
void callback(const geometry_msgs::PoseArray::ConstPtr& msg)
{
  cout << "Callback triggered" << endl;
  double tf = 3;
  int N = (int) 50*tf;
  int epochs = 10;
  Body3dState x0;
  //Read from TF
  geometry_msgs::TransformStamped temp;
  try {
    temp = tfBuffer.lookupTransform(world_name,matrice_name,ros::Time(0),ros::Duration(5.0));
  } catch (tf2::TransformException &ex) {
    ROS_WARN("Could not find the transform");
    return;
  }
  x0.p << temp.transform.translation.x, temp.transform.translation.y, temp.transform.translation.z;
  quat2Matrix(temp.transform.rotation.x,temp.transform.rotation.y,temp.transform.rotation.z,temp.transform.rotation.w,x0.R);
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
    goal[ii].p = (1 - filter_alpha)*last_pos[ii] + filter_alpha*new_pos;
    last_pos[ii] = goal[ii].p;
    last_quat[ii] = last_quat[ii].slerp(matrix2Quat(new_rot),filter_alpha);
    quat2Matrix(last_quat[ii],goal[ii].R);
    goal[ii].v << 0, 0, 0;
    goal[ii].w << 0, 0, 0;
    
    temp.header.stamp = ros::Time::now();
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
  double yawgain = 0;
  double rpgain = 0;
  double dir_gain = 0;
  vector<Body3dState> xs(N+1);
  vector<Vector4d> us(N);
  solver_process_goal(N, tf, epochs, x0, goal[0], goal[1], goal[2],q, qf, r, yawgain, rpgain, dir_gain, xs, us);
  nav_msgs::Path pa;
  pa.poses.clear();
  pa.header.stamp = ros::Time::now();
  pa.header.frame_id = world_name;
  for (int ii = 0; ii < N+1; ++ii) {
    geometry_msgs::PoseStamped p;
    p.header.stamp = ros::Time::now();
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
  }
  pub.publish(pa);
  cout << "Callback Ended" << endl;
  first_call = false;
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
  std::string path_topic, goal_topic;
  if (!(n.getParam("path_topic",path_topic))){
    path_topic = "path";
  }
  if (!(n.getParam("goal_topic",goal_topic))){
    goal_topic = "goal_points";
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
  cout << "Publishing to " << path_topic << endl;
  cout << "Subscribing to " << goal_topic << endl;
  cout << "Matrice Name set to " << matrice_name << endl;
  cout << "World Name set to " << world_name << endl;
  cout << "Goal Name set to " << goal_name << endl;
  cout << "Filter Alpha set to " << filter_alpha << endl;
  pub = n.advertise<nav_msgs::Path>(path_topic,1000);
  sub = n.subscribe(goal_topic,1000,&TrajectoryNode::callback,this);
  first_call = true;
}

tf2_ros::Buffer tfBuffer;
tf2_ros::TransformListener lis;
tf2_ros::TransformBroadcaster br;
ros::NodeHandle n;
ros::Publisher pub;
ros::Subscriber sub;
std::string world_name;
std::string matrice_name;
std::string goal_name;
double filter_alpha;
vector<Vector3d> last_pos;
vector<tf2::Quaternion> last_quat;
bool first_call;
};
/*The main function
Builds an instance of TrajectoryNode and then spins.
*/
int main(int argc, char **argv)
{
  cout << "Init" << endl;
  ros::init(argc,argv, "Trajectory_Node");
  cout << "Constructor" << endl;
  TrajectoryNode tn;
  cout << "Spin" << endl;
  ros::spin();
  return 0;
}
