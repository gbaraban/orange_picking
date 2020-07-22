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
//#include "gcop/quat.h"

using namespace std;
using namespace Eigen;
using namespace gcop;

class TrajectoryNode
{

typedef Ddp<Body3dState, 12, 4> HrotorDdp;
typedef ConstraintCost<Body3dState, 12, 4, Dynamic, 3> DirectionConstraintCost;

public:

//Params params;
void solver_process_goal(int N, double tf, int epochs, Body3dState x0,
     Body3dState goal1, Body3dState goal2, Body3dState goal3,
     Vector12d q, Vector12d qf, Vector4d r, double yawgain, double rpgain,double dir_gain,
     vector<Body3dState> &xout, vector<Vector4d> &us)
{
/*  cout << "x0: pos: " << x0.p[0] << " " << x0.p[1] << " " << x0.p[2] << endl;
  cout << "x0: Rot: " << endl << x0.R(0,0) << " " << x0.R(0,1) << " " << x0.R(0,2) << endl;
  cout << x0.R(1,0) << " " << x0.R(1,1) << " " << x0.R(1,2) << endl;
  cout << x0.R(2,0) << " " << x0.R(2,1) << " " << x0.R(2,2) << endl;
  cout << "goal1: pos: " << goal1.p[0] << " " << goal1.p[1] << " " << goal1.p[2] << endl;
  cout << "goal1: Rot: " << endl << goal1.R(0,0) << " " << goal1.R(0,1) << " " << goal1.R(0,2) << endl;
  cout << goal1.R(1,0) << " " << goal1.R(1,1) << " " << goal1.R(1,2) << endl;
  cout << goal1.R(2,0) << " " << goal1.R(2,1) << " " << goal1.R(2,2) << endl;
  cout << "goal2: pos: " << goal2.p[0] << " " << goal2.p[1] << " " << goal2.p[2] << endl;
  cout << "goal2: Rot: " << endl << goal2.R(0,0) << " " << goal2.R(0,1) << " " << goal2.R(0,2) << endl;
  cout << goal2.R(1,0) << " " << goal2.R(1,1) << " " << goal2.R(1,2) << endl;
  cout << goal2.R(2,0) << " " << goal2.R(2,1) << " " << goal2.R(2,2) << endl;
  cout << "goal3: pos: " << goal3.p[0] << " " << goal3.p[1] << " " << goal3.p[2] << endl;
  cout << "goal3: Rot: " << endl << goal3.R(0,0) << " " << goal3.R(0,1) << " " << goal3.R(0,2) << endl;
  cout << goal3.R(1,0) << " " << goal3.R(1,1) << " " << goal3.R(1,2) << endl;
  cout << goal3.R(2,0) << " " << goal3.R(2,1) << " " << goal3.R(2,2) << endl;*/
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
  tf2::Quaternion temp_quat(temp.transform.rotation.x,temp.transform.rotation.y,temp.transform.rotation.z,temp.transform.rotation.w);
  tf2::Matrix3x3 temp_mat(temp_quat);
  //double m[16];
  //quat.ToSE3(m);
  /*cout << "x0 temp_mat: " << endl;
  for (int ii = 0; ii < 3; ++ ii) {
    for (int jj = 0; jj < 3; ++ jj) {
      cout << temp_mat[ii][jj] << " ";
    }
    cout << endl;
  }*/
  //x0.R << m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]; 
  x0.R << temp_mat[0][0], temp_mat[0][1], temp_mat[0][2],
          temp_mat[1][0], temp_mat[1][1], temp_mat[1][2],
          temp_mat[2][0], temp_mat[2][1], temp_mat[2][2]; 
  Body3dState goal[3];
  for (int ii = 0; ii < 3; ++ii){
    Vector3d local_p;
    local_p << msg->poses[ii].position.x, msg->poses[ii].position.y, msg->poses[ii].position.z;
    //tf2::convert(msg->poses[ii].orientation,temp_quat);
    temp_quat = tf2::Quaternion(msg->poses[ii].orientation.x,msg->poses[ii].orientation.y,msg->poses[ii].orientation.z,msg->poses[ii].orientation.w);
    //Quat(msg->poses[ii].orientation.w,msg->poses[ii].orientation.x,msg->poses[ii].orientation.y,msg->poses[ii].orientation.z);
    //quat.ToSE3(m);
    temp_mat = tf2::Matrix3x3(temp_quat);
    Matrix3d local_R;
    local_R << temp_mat[0][0], temp_mat[0][1], temp_mat[0][2],
               temp_mat[1][0], temp_mat[1][1], temp_mat[1][2],
               temp_mat[2][0], temp_mat[2][1], temp_mat[2][2]; 
    //local_R << m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10];
    goal[ii].p = x0.R*local_p + x0.p;//TODO: double check this.  maybe local R needs to be inverted?
    goal[ii].R = x0.R*local_R;
    goal[ii].v << 0, 0, 0;
    goal[ii].w << 0, 0, 0;
    temp.header.stamp = ros::Time::now();
    temp.header.frame_id = matrice_name;
    temp.child_frame_id = goal_name + std::to_string(ii);
    temp.transform.translation.x = msg->poses[ii].position.x;
    temp.transform.translation.y = msg->poses[ii].position.y;
    temp.transform.translation.z = msg->poses[ii].position.z;
    temp.transform.rotation.x = msg->poses[ii].orientation.x;
    temp.transform.rotation.y = msg->poses[ii].orientation.y;
    temp.transform.rotation.z = msg->poses[ii].orientation.z;
    temp.transform.rotation.w = msg->poses[ii].orientation.w;
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
  //cout << "Solved: pos: " << xs[0].p[0] << " " << xs[0].p[1] << " " << xs[0].p[2] << endl;
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
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
        double temp_val = xs[ii].R(jj,kk);
        temp_mat[jj][kk] = temp_val;
      }
    }
    temp_mat.getRotation(temp_quat);
    p.pose.orientation.x = temp_quat.getX();
    p.pose.orientation.y = temp_quat.getY();
    p.pose.orientation.z = temp_quat.getZ();
    p.pose.orientation.w = temp_quat.getW();
    pa.poses.push_back(p);
  }
  //cout << "Publishing pa" << endl;
  pub.publish(pa);
  cout << "Callback Ended" << endl;
}

TrajectoryNode(): lis(tfBuffer)
{
  std::string path_topic, goal_topic;
  if (!(n.getParam("path_topic",path_topic))){
    path_topic = "path";
  }
  if (!(n.getParam("goal_topic",goal_topic))){
    goal_topic = "goal";
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
  cout << "Publishing to " << path_topic << endl;
  cout << "Subscribing to " << goal_topic << endl;
  pub = n.advertise<nav_msgs::Path>(path_topic,1000);
  sub = n.subscribe(goal_topic,1000,&TrajectoryNode::callback,this);
  //lis = tf2_ros::TransformListener(tfBuffer);
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
};

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
