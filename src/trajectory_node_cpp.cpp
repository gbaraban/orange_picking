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
//#include <Matrix3x3.h>
#include <string>
#include "gcop/quat.h"

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
  cout << "N: " << N << " tf: " << tf << " epochs: "  << epochs << endl;
  cout << "x0: pos: " << x0.p[0] << " " << x0.p[1] << " " << x0.p[2] << endl;
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
  cout << goal3.R(2,0) << " " << goal3.R(2,1) << " " << goal3.R(2,2) << endl;
  cout << "Q: " << q[0] << " " << q[1] << " " << q[2] << endl << q[3] << " " << q[4] << " " << q[5] << endl << q[6] << " " << q[7] << " " << q[8] << endl << q[9] << " " << q[10] << " " << q[11] << endl; 
  cout << "Qf: " << qf[0] << " " << qf[1] << " " << qf[2] << endl << qf[3] << " " << qf[4] << " " << qf[5] << endl << qf[6] << " " << qf[7] << " " << qf[8] << endl << qf[9] << " " << qf[10] << " " << qf[11] << endl; 
  cout << "R: " << r[0] << " " << r[1] << " " << r[2] << " " << r[3] << endl;
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
  cout << "Looking Up" << world_name << "-> " << matrice_name << endl;
  try {
    temp = tfBuffer.lookupTransform(world_name,matrice_name,ros::Time(0),ros::Duration(5.0));
  } catch (tf2::TransformException &ex) {
    ROS_WARN("Could not find the transform");
    return;
  }
  cout << "Looked Up" << endl;
  x0.p << temp.transform.translation.x, temp.transform.translation.y, temp.transform.translation.z;
  Quat quat(temp.transform.rotation.w,temp.transform.rotation.x,temp.transform.rotation.y,temp.transform.rotation.z);
  double m[16];
  quat.ToSE3(m);
  x0.R << m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]; 
  Body3dState goal[3];
  for (int ii = 0; ii < 3; ++ii){
    cout << "2" << endl;
    Vector3d local_p;
    local_p << msg->poses[ii].position.x, msg->poses[ii].position.y, msg->poses[ii].position.z;
    quat = Quat(msg->poses[ii].orientation.w,msg->poses[ii].orientation.x,msg->poses[ii].orientation.y,msg->poses[ii].orientation.z);
    quat.ToSE3(m);
    Matrix3d local_R;
    local_R << m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10];
    goal[ii].p = x0.R*local_p + x0.p;
    goal[ii].R = x0.R*local_R;
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
    cout << "Publishing Goal " << ii << endl;
    br.sendTransform(temp);
  }
  Vector12d q;
  q << 0,0,0,0,0,0,10,10,10,10,10,10;
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
  cout << "Solved: pos: " << xs[0].p[0] << " " << xs[0].p[1] << " " << xs[0].p[2] << endl;
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
        int temp_idx = 4*jj + kk;
        double temp_val = xs[ii].R(jj,kk);
        m[temp_idx] = temp_val;
      }
      m[4*jj + 3] = 0;
    }
    m[15] = 0;
    quat.FromSE3(m);
    p.pose.orientation.x = quat.qx;
    p.pose.orientation.y = quat.qy;
    p.pose.orientation.z = quat.qz;
    p.pose.orientation.w = quat.qw;
    pa.poses.push_back(p);
  }
  cout << "Publishing pa" << endl;
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
