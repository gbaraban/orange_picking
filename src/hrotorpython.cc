#include <Python.h>
#include <iostream>
#include "utils.h"
//#include "params.h"
#include "body3dcost.h"
#include "body3dwaypointcost.h"
#include "multicost.h"
#include "cylinder.h"
#include "groundplane.h"
#include "yawvelocityconstraint.h"
#include "constraintcost.h"
#include "yawcost.h"
#include "rpcost.h"
#include "direction_constraint.h"
#include "hrotor.h"
#include "ddp.h"
#include "so3.h"
#include <Eigen/Geometry>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace gcop;

typedef Ddp<Body3dState, 12, 4> HrotorDdp;
typedef ConstraintCost<Body3dState, 12, 4, Dynamic, 3> DirectionConstraintCost;

//Params params;
void solver_process(int N, double tf, int epochs, Body3dState x0,
		    Vector3d xfp, double yawf, Vector3d cyl_o, double cyl_r, double cyl_h,
                    Vector12d q, Vector12d qf, Vector4d r, 
		    double yawgain, double rpgain,double dir_gain,
                    double stiffness, double stiff_mult, vector<Body3dState> &xout, vector<Vector4d> &us)
{

  //Parameters
  double h = tf/N;
  
  //System
  Hrotor sys;

  //Costs
  MultiCost<Body3dState, 12, 4> cost(sys, tf);

  //Quadratic Cost
  Body3dState xf;
  xf.Clear();
  xf.p = xfp;
  SO3::Instance().q2g(xf.R, Vector3d(0,0,yawf)); 

  Body3dCost<4> pathcost(sys, tf, xf);
  for (int i = 0; i < 12; ++i)
  {
    pathcost.Q(i,i) = q[i];
    pathcost.Qf(i,i) = qf[i];
  }
  for (int i = 0; i < 4; ++i)
  {
    pathcost.R(i,i) = r[i];
  }
  cost.costs.push_back(&pathcost);
  
  //Cylinder Cost
  Cylinder<Body3dState, 12, 4> cyl(cyl_o, cyl_r, cyl_h, 0.0); //0 is the collision radius
  ConstraintCost<Body3dState, 12, 4> cylcost(sys, tf, cyl);
  cost.costs.push_back(&cylcost);

  //Ground Cost
  GroundPlane<Body3dState, 12, 4> gp(0);
  ConstraintCost<Body3dState, 12, 4> gpcost(sys,tf,gp);
  cost.costs.push_back(&gpcost);

  //Yaw Cost
  //YawVelocityConstraint<Body3dState, 12, 4> yaw_con;
  //ConstraintCost<Body3dState, 12, 4> yawcost(sys,tf,yaw_con);
  YawCost<Body3dState, 12, 4> yawcost(sys,tf,xfp);
  yawcost.gain = yawgain;
  cost.costs.push_back(&yawcost);

  //Roll Pitch Cost
  RPCost<Body3dState, 12, 4> rpcost(sys,tf);
  rpcost.gain = rpgain;
  cost.costs.push_back(&rpcost);

  //Direction Constraint
  DirectionConstraint<4> direction_constraint(Vector3d(1,0,0),xfp);
  DirectionConstraintCost direction_cost(sys,tf,direction_constraint);
  direction_cost.b = dir_gain;
  cost.costs.push_back(&direction_cost);

  // Times
  vector<double> ts(N+1);
  for (int k = 0; k <= N; ++k)
    ts[k] = k*h;

  // States
  xout.resize(N+1);
  xout[0].Clear();
  xout[0] = x0;
  //SO3::Instance().q2g(xout[0].R, Vector3d(0,0,yaw0));    

  // initial controls (e.g. hover at one place)
  //vector<Vector4d> us(N);
  //vector<Body3dState> xds(N+1);
  us.resize(N);
  vector<Vector4d> uds(N);
  for (int i = 0; i < N; ++i) {
    us[i].head(3).setZero();
    us[i][3] = 9.81*sys.m;
    uds[i].head(3).setZero();
    uds[i][3] = 9.81*sys.m;
    //xds[i].p = xfp;
  }
  //xds[N].p = xfp;
  //xds[N].R = final_R;
  pathcost.SetReference(NULL,&uds);

  HrotorDdp ddp(sys, cost, ts, xout, us);
  ddp.mu = 1;
  ddp.debug = false;

 // struct timeval timer;
 // timer_start(timer);
  for (double temp_b = 0; temp_b < stiffness; temp_b = temp_b*stiff_mult) {
    cylcost.b = temp_b;
    gpcost.b = temp_b;
    //yawcost.b = temp_b;
    for (int ii = 0; ii < epochs; ++ii) {
      ddp.Iterate();
     //cout << "Stiffness: " << cylcost.b << " Iteration Num: " << ii << " DDP V: " << ddp.V << endl;
     // long te = timer_us(timer);
     // if (te > time_limit) break;
    }
    if (temp_b < 1) {
      temp_b = 1;
    }
  }
}

static PyObject *
gcophrotor_trajgen_R(PyObject *self, PyObject *args)
{
  int N;
  double tf;
  int epochs;
  double x0x, x0y, x0z;
  double R01, R02, R03;
  double R04, R05, R06;
  double R07, R08, R09;
  double v0x, v0y, v0z;
  double w0x, w0y, w0z;
  double xfx, xfy, xfz, yawf;
  double cx, cy, cz;
  double cyl_r;
  double cyl_h;
  Vector12d q;
  Vector12d qf;
  Vector4d r;
  double yawgain;
  double rpgain;
  double dir_gain;
  double stiffness;
  double stiff_mult;
  if (!PyArg_ParseTuple(args, "idi(ddd)(ddddddddd)(ddd)(ddd)(ddd)d(ddd)dd(dddddddddddd)(dddddddddddd)(dddd)ddddd",
        &N, &tf, &epochs, &x0x, &x0y, &x0z,
        &R01, &R02, &R03,
        &R04, &R05, &R06,
        &R07, &R08, &R09,
	&v0x,&v0y,&v0z,&w0x,&w0y,&w0z,
        &xfx, &xfy, &xfz, &yawf,
        &cx, &cy, &cz, &cyl_r, &cyl_h,
        &(q[0]), &(q[1]), &(q[2]), &(q[3]),
        &(q[4]), &(q[5]), &(q[6]), &(q[7]),
        &(q[8]), &(q[9]), &(q[10]), &(q[11]),
        &(qf[0]), &(qf[1]), &(qf[2]), &(qf[3]),
        &(qf[4]), &(qf[5]), &(qf[6]), &(qf[7]),
        &(qf[8]), &(qf[9]), &(qf[10]), &(qf[11]),
        &(r[0]), &(r[1]), &(r[2]), &(r[3]),&yawgain,&rpgain,&dir_gain,
        &stiffness, &stiff_mult)) {
    cout << "Parse Failed" << endl;
    return NULL;
  }
  vector<Body3dState> xs(N+1);
  Body3dState x0;
  x0.p << x0x, x0y, x0z;
  x0.R << R01, R02, R03, R04, R05, R06, R07, R08, R09;
  x0.v << v0x, v0y, v0z;
  x0.w << w0x, w0y, w0z;
  Vector3d xfp(xfx,xfy,xfz);
  Vector3d cyl_o(cx,cy,cz);
  vector<Vector4d> us(N);
  solver_process(N, tf, epochs, x0, xfp, yawf,
                 cyl_o, cyl_r, cyl_h, q,qf,r,yawgain,rpgain,dir_gain,
                 stiffness, stiff_mult, xs, us);
  //Construct return object for xs
  PyObject* xsObj = PyList_New(xs.size());
  if (!xsObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < xs.size(); ++ii) {
    Vector3d pos = xs[ii].p;
    Matrix3d rot = xs[ii].R;
    Vector3d v = xs[ii].v;
    Vector3d w = xs[ii].w;
    PyObject *tuple = Py_BuildValue("(ddd)((ddd),(ddd),(ddd))(ddd)(ddd)",
                                     pos(0),pos(1),pos(2),
                                     rot(0,0),rot(0,1),rot(0,2),
                                     rot(1,0),rot(1,1),rot(1,2),
                                     rot(2,0),rot(2,1),rot(2,2),
				     v(0),v(1),v(2),
				     w(0),w(1),w(2));
    if (!tuple) {
      Py_DECREF(xsObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(xsObj, ii, tuple);
  }
  //Construct return object for us
  PyObject* usObj = PyList_New(us.size());
  if (!usObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < us.size(); ++ii) {
    Vector4d u = us[ii];
    PyObject *tuple = Py_BuildValue("dddd",u(0),u(1),u(2),u(3));
    if (!tuple) {
      Py_DECREF(usObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(usObj, ii, tuple);
  }
  //Combine return object
  PyObject* retObj = PyList_New(2);
  if (!retObj) throw logic_error("Failed to make list Object");
  PyList_SET_ITEM(retObj,0,xsObj);
  PyList_SET_ITEM(retObj,1,usObj);
  return retObj;
}

//Params params;
void solver_process_goal(int N, double tf, int epochs, Body3dState x0,
     Body3dState goal1, Body3dState goal2, Body3dState goal3,
     Vector12d q, Vector12d qf, Vector4d r, double yawgain, double rpgain,double dir_gain,
     vector<Body3dState> &xout, vector<Vector4d> &us)
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
  //cout << "Waypoint Cost Time List Size: " << waypointcost.time_list.size() << endl;
  //for (int i = 0; i < pathcost.cost_list.size(); ++i) {
    for (int j = 0; j < 12; ++j)
    {
      //pathcost.cost_list[i].Q(j,j) = q[j];
      //pathcost.cost_list[i].Qf(j,j) = qf[j];
      pathcost.Q(j,j) = q[j];
      waypointcost.Q(j,j) = qf[j];
    }
    for (int j = 0; j < 4; ++j)
    {
      //pathcost.cost_list[i].R(j,j) = r[j];
      pathcost.R(j,j) = r[j];
    }
  //}
  cost.costs.push_back(&pathcost);
  cost.costs.push_back(&waypointcost);
  //Yaw Cost
  //YawVelocityConstraint<Body3dState, 12, 4> yaw_con;
  //ConstraintCost<Body3dState, 12, 4> yawcost(sys,tf,yaw_con);
  //YawCost<Body3dState, 12, 4> yawcost(sys,tf,xfp);
  //yawcost.gain = yawgain;
  //cost.costs.push_back(&yawcost);


  // Times
  vector<double> ts(N+1);
  for (int k = 0; k <= N; ++k)
    ts[k] = k*h;

  // States
  xout.resize(N+1);
  xout[0].Clear();
  xout[0] = x0;
  //SO3::Instance().q2g(xout[0].R, Vector3d(0,0,yaw0));    

  // initial controls (e.g. hover at one place)
  us.resize(N);
  //vector<Body3dState> xds(N+1);
  vector<Vector4d> uds(N);
  double third_N =((double) N)/goals.size();
  for (int i = 0; i < N; ++i) {
    us[i].head(3).setZero();
    us[i][3] = 9.81*sys.m;
    uds[i].head(3).setZero();
    uds[i][3] = 9.81*sys.m;
    //int goal_num = (int)(((double)i)/third_N);
    //xds[i] = goals[goal_num];
  }
  //xds[N] = goals[2];//.p = xfp;
  //xds[N].R = final_R;
  //for (int ii = 0; ii < pathcost.cost_list.size(); ++ii) {
  //  pathcost.cost_list[ii].SetReference(NULL,&uds);
  //}
  pathcost.SetReference(NULL,&uds);

  HrotorDdp ddp(sys, cost, ts, xout, us);
  ddp.mu = 1;
  ddp.debug = false;

 // struct timeval timer;
 // timer_start(timer);
    //yawcost.b = temp_b;
  //cout << "Pre For DDP" << endl;
  for (int ii = 0; ii < epochs; ++ii) {
    ddp.Iterate();
    cout << " Iteration Num: " << ii << " DDP V: " << ddp.V << endl;
   // long te = timer_us(timer);
   // if (te > time_limit) break;
  }
}

static PyObject *
gcophrotor_trajgen_goal(PyObject *self, PyObject *args)
{
  
  int N;
  double tf;
  int epochs;
  double x0x, x0y, x0z;
  double R01, R02, R03;
  double R04, R05, R06;
  double R07, R08, R09;
  double v0x, v0y, v0z;
  double w0x, w0y, w0z;
  double g1x, g1y, g1z;
  double R11, R12, R13;
  double R14, R15, R16;
  double R17, R18, R19;
  double g2x, g2y, g2z;
  double R21, R22, R23;
  double R24, R25, R26;
  double R27, R28, R29;
  double g3x, g3y, g3z;
  double R31, R32, R33;
  double R34, R35, R36;
  double R37, R38, R39;
  Vector12d q;
  Vector12d qf;
  Vector4d r;
  double yawgain;
  double rpgain;
  double dir_gain;
  if (!PyArg_ParseTuple(args, "idi((ddd)(ddddddddd)(ddd)(ddd))((ddd)(ddddddddd))((ddd)(ddddddddd))((ddd)(ddddddddd))(dddddddddddd)(dddddddddddd)(dddd)ddd", 
        &N, &tf, &epochs,
	&x0x, &x0y, &x0z, &R01, &R02, &R03, &R04, &R05, &R06, &R07, &R08, &R09, 
	&v0x,&v0y,&v0z,&w0x,&w0y,&w0z,
	&g1x, &g1y, &g1z, &R11, &R12, &R13, &R14, &R15, &R16, &R17, &R18, &R19, 
	&g2x, &g2y, &g2z, &R21, &R22, &R23, &R24, &R25, &R26, &R27, &R28, &R29, 
	&g3x, &g3y, &g3z, &R31, &R32, &R33, &R34, &R35, &R36, &R37, &R38, &R39, 
        &(q[0]), &(q[1]), &(q[2]), &(q[3]),
        &(q[4]), &(q[5]), &(q[6]), &(q[7]),
        &(q[8]), &(q[9]), &(q[10]), &(q[11]),
        &(qf[0]), &(qf[1]), &(qf[2]), &(qf[3]),
        &(qf[4]), &(qf[5]), &(qf[6]), &(qf[7]),
        &(qf[8]), &(qf[9]), &(qf[10]), &(qf[11]),
        &(r[0]), &(r[1]), &(r[2]), &(r[3]),&yawgain,&rpgain,&dir_gain)) {
    cout << "Parse Failed" << endl;
    return NULL;
  }
  vector<Body3dState> xs(N+1);
  vector<Vector4d> us(N);
  Body3dState x0;
  x0.p << x0x, x0y, x0z;
  x0.R << R01, R02, R03, R04, R05, R06, R07, R08, R09;
  x0.v << v0x, v0y, v0z;
  x0.w << w0x, w0y, w0z;
  Body3dState goal1;
  goal1.p << g1x, g1y, g1z;
  goal1.R << R11, R12, R13, R14, R15, R16, R17, R18, R19;
  Body3dState goal2;
  goal2.p << g2x, g2y, g2z;
  goal2.R << R21, R22, R23, R24, R25, R26, R27, R28, R29;
  Body3dState goal3;
  goal3.p << g3x, g3y, g3z;
  goal3.R << R31, R32, R33, R34, R35, R36, R37, R38, R39;
  solver_process_goal(N, tf, epochs, x0, goal1,goal2,goal3,
                 q,qf,r,yawgain,rpgain,dir_gain,
                 xs, us);
  //Construct return object for xs
  PyObject* xsObj = PyList_New(xs.size());
  if (!xsObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < xs.size(); ++ii) {
    Vector3d pos = xs[ii].p;
    Matrix3d rot = xs[ii].R;
    Vector3d v = xs[ii].v;
    Vector3d w = xs[ii].w;
    PyObject *tuple = Py_BuildValue("(ddd)((ddd),(ddd),(ddd))(ddd)(ddd)",
                                     pos(0),pos(1),pos(2),
                                     rot(0,0),rot(0,1),rot(0,2),
                                     rot(1,0),rot(1,1),rot(1,2),
                                     rot(2,0),rot(2,1),rot(2,2),
				     v(0),v(1),v(2),
				     w(0),w(1),w(2));
    if (!tuple) {
      Py_DECREF(xsObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(xsObj, ii, tuple);
  }
  //Construct return object for us
  PyObject* usObj = PyList_New(us.size());
  if (!usObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < us.size(); ++ii) {
    Vector4d u = us[ii];
    PyObject *tuple = Py_BuildValue("dddd",u(0),u(1),u(2),u(3));
    if (!tuple) {
      Py_DECREF(usObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(usObj, ii, tuple);
  }
  //Combine return object
  PyObject* retObj = PyList_New(2);
  if (!retObj) throw logic_error("Failed to make list Object");
  PyList_SET_ITEM(retObj,0,xsObj);
  PyList_SET_ITEM(retObj,1,usObj);
  return retObj;
}

static PyObject *
gcophrotor_trajgen(PyObject *self, PyObject *args)
{
  
  int N;
  double tf;
  int epochs;
  double x0x, x0y, x0z, yaw0;
  double xfx, xfy, xfz, yawf;
  double cx, cy, cz;
  double cyl_r;
  double cyl_h;
  Vector12d q;
  Vector12d qf;
  Vector4d r;
  double yawgain;
  double rpgain;
  double dir_gain;
  double stiffness;
  double stiff_mult;
  if (!PyArg_ParseTuple(args, "idi(ddd)d(ddd)d(ddd)dd(dddddddddddd)(dddddddddddd)(dddd)ddddd", 
        &N, &tf, &epochs, &x0x, &x0y, &x0z, &yaw0, &xfx, &xfy, &xfz, &yawf,
        &cx, &cy, &cz, &cyl_r, &cyl_h, 
        &(q[0]), &(q[1]), &(q[2]), &(q[3]),
        &(q[4]), &(q[5]), &(q[6]), &(q[7]),
        &(q[8]), &(q[9]), &(q[10]), &(q[11]),
        &(qf[0]), &(qf[1]), &(qf[2]), &(qf[3]),
        &(qf[4]), &(qf[5]), &(qf[6]), &(qf[7]),
        &(qf[8]), &(qf[9]), &(qf[10]), &(qf[11]),
        &(r[0]), &(r[1]), &(r[2]), &(r[3]),&yawgain,&rpgain,&dir_gain,
        &stiffness, &stiff_mult)) {
    cout << "Parse Failed" << endl;
    return NULL;
  }
  vector<Body3dState> xs(N+1);
  Body3dState x0;
  x0.p << x0x,x0y,x0z;
  x0.R << cos(yaw0), -sin(yaw0), 0, sin(yaw0), cos(yaw0), 0, 0, 0, 1;
  Vector3d xfp(xfx,xfy,xfz);
  Vector3d cyl_o(cx,cy,cz);
  vector<Vector4d> us(N);
  solver_process(N, tf, epochs, x0, xfp, yawf,
                 cyl_o, cyl_r, cyl_h, q,qf,r,yawgain,rpgain,dir_gain,
                 stiffness, stiff_mult, xs, us);
  //Construct return object for xs
  PyObject* xsObj = PyList_New(xs.size());
  if (!xsObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < xs.size(); ++ii) {
    Vector3d pos = xs[ii].p;
    Matrix3d rot = xs[ii].R;
    Vector3d v = xs[ii].v;
    Vector3d w = xs[ii].w;
    PyObject *tuple = Py_BuildValue("(ddd)((ddd),(ddd),(ddd))(ddd)(ddd)",
                                     pos(0),pos(1),pos(2),
                                     rot(0,0),rot(0,1),rot(0,2),
                                     rot(1,0),rot(1,1),rot(1,2),
                                     rot(2,0),rot(2,1),rot(2,2),
				     v(0),v(1),v(2),
				     w(0),w(1),w(2));
    if (!tuple) {
      Py_DECREF(xsObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(xsObj, ii, tuple);
  }
  //Construct return object for us
  PyObject* usObj = PyList_New(us.size());
  if (!usObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < us.size(); ++ii) {
    Vector4d u = us[ii];
    PyObject *tuple = Py_BuildValue("dddd",u(0),u(1),u(2),u(3));
    if (!tuple) {
      Py_DECREF(usObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(usObj, ii, tuple);
  }
  //Combine return object
  PyObject* retObj = PyList_New(2);
  if (!retObj) throw logic_error("Failed to make list Object");
  PyList_SET_ITEM(retObj,0,xsObj);
  PyList_SET_ITEM(retObj,1,usObj);
  return retObj;
}

static PyObject *
gcophrotor_dynamic_step(PyObject *self, PyObject *args)
{
  double t;
  double R01, R02, R03;
  double R04, R05, R06;
  double R07, R08, R09;
  Body3dState x;
  Vector4d u;
  if (!PyArg_ParseTuple(args, "d(ddd)(ddddddddd)(ddd)(ddd)(dddd)", 
        &t, &(x.p[0]), &(x.p[1]),&(x.p[2]),
        &R01, &R02, &R03, 
        &R04, &R05, &R06, 
        &R07, &R08, &R09, 
	&(x.v[0]), &(x.v[1]), &(x.v[2]),
	&(x.w[0]), &(x.w[1]), &(x.w[2]),
	&(u[0]), &(u[1]), &(u[2]), &(u[3]))) {
    cout << "Parse Failed" << endl;
    return NULL;
  }
  x.R << R01, R02, R03,
        R04, R05, R06,
        R07, R08, R09;
  Hrotor sys;
  Body3dState xb;
  sys.Step(xb,t, x, u,t);
  PyObject *tuple = Py_BuildValue("(ddd)(ddddddddd)(ddd)(ddd)",
		  xb.p(0),xb.p(1),xb.p(2),
                  xb.R(0,0),xb.R(0,1),xb.R(0,2),
                  xb.R(1,0),xb.R(1,1),xb.R(1,2),
                  xb.R(2,0),xb.R(2,1),xb.R(2,2),
		  xb.v(0),xb.v(1),xb.v(2),
		  xb.w(0),xb.w(1),xb.w(2));
  return tuple;
}
static PyMethodDef pyMethods[] = {
  {"trajgen", gcophrotor_trajgen, METH_VARARGS, "Generate a trajectory."},
  {"trajgen_R", gcophrotor_trajgen_R, METH_VARARGS, "Generate a trajectory with R fully defined."},
  {"trajgen_goal", gcophrotor_trajgen_goal, METH_VARARGS, "Generate a trajectory from waypoints (with R fully defined)."},
  {"dynamic_step", gcophrotor_dynamic_step, METH_VARARGS, "Perform a single dynamic step."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gcophrotor_definition = {
  PyModuleDef_HEAD_INIT,
  "gcophrotor",
  "GCOP with Hrotor",
  -1,
  pyMethods};

PyMODINIT_FUNC PyInit_gcophrotor(void)
{
  //(void) PyInitModule("GCOP Hrotor", pyMethods);
  Py_Initialize();
  return PyModule_Create(&gcophrotor_definition);
}
/*
int main(int argc, char** argv)
{
  int N = 64;
  double tf = 20;
  int epochs = 100;
  //double time_limit = 200;
  Vector3d x0(5,5,2);
  Vector3d xfp(-5,-2,5);
  Vector3d cyl_o(0,0,0);
  double cyl_r = 2;
  double cyl_h = 10;
  double stiffness = 1e6;
  double stiff_mult = 2;
  vector<Body3dState> xs(N+1);
  solver_process(N, tf, epochs, x0, xfp, 
                 cyl_o, cyl_r, cyl_h, 
                 stiffness, stiff_mult, xs);
  cout << "Trajectory" << endl;
  for (int i = 0; i <= N; ++i) {
     cout << xs[i].p << endl;
  }
  return 0;

//  Py_SetProgramName(argv[0]);
  Py_Initialize();
  initgcophrotor();
  return 0;
}*/
