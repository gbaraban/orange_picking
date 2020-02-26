#include <Python.h>
#include <iostream>
#include "utils.h"
//#include "params.h"
#include "body3dcost.h"
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

using namespace std;
using namespace Eigen;
using namespace gcop;

typedef Ddp<Body3dState, 12, 4> HrotorDdp;
typedef ConstraintCost<Body3dState, 12, 4, Dynamic, 3> DirectionConstraintCost;

//Params params;
void solver_process(int N, double tf, int epochs, Vector3d x0, Matrix3d R0, Vector3d xfp, double yawf,
     Vector3d cyl_o, double cyl_r, double cyl_h,
     Vector12d q, Vector12d qf, Vector4d r, double yawgain, double rpgain,double dir_gain,
      double stiffness, double stiff_mult, vector<Body3dState> &xout)
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
  xout[0].p = x0;
  xout[0].R = R0;
  //SO3::Instance().q2g(xout[0].R, Vector3d(0,0,yaw0));    

  // initial controls (e.g. hover at one place)
  vector<Vector4d> us(N);
  //vector<Body3dState> xds(N+1);
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
      /*for (int jj = 0; jj < N; ++jj) {
        Vector3d temp(0,0,atan2(xout[jj].v[1],xout[jj].v[0]));//yaw from velocity
        SO3::Instance().q2g(xds[jj].R,temp);
      }*/
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
        &N, &tf, &epochs, &x0x, &x0y, &x0z, 
        &R01, &R02, &R03, 
        &R04, &R05, &R06, 
        &R07, &R08, R09, 
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
  Vector3d x0(x0x,x0y,x0z);
  Vector3d xfp(xfx,xfy,xfz);
  Vector3d cyl_o(cx,cy,cz);
  Matrix3d R0;
  R0 << R01, R02, R03,
        R04, R05, R06,
        R07, R08, R09;
  solver_process(N, tf, epochs, x0, R0, xfp, yawf,
                 cyl_o, cyl_r, cyl_h, q,qf,r,yawgain,rpgain,dir_gain,
                 stiffness, stiff_mult, xs);
  //Construct return object
  PyObject* listObj = PyList_New(xs.size());
  if (!listObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < xs.size(); ++ii) {
    Vector3d pos = xs[ii].p;
    Matrix3d rot = xs[ii].R;
    PyObject *tuple = Py_BuildValue("(ddd)((ddd),(ddd),(ddd))",
                                     pos(0),pos(1),pos(2),
                                     rot(0,0),rot(0,1),rot(0,2),
                                     rot(1,0),rot(1,1),rot(1,2),
                                     rot(2,0),rot(2,1),rot(2,2));
    if (!tuple) {
      Py_DECREF(listObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(listObj, ii, tuple);
  }
  return listObj;
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
  Vector3d x0(x0x,x0y,x0z);
  Vector3d xfp(xfx,xfy,xfz);
  Vector3d cyl_o(cx,cy,cz);
  Matrix3d R0;
  R0 << cos(yaw0), -sin(yaw0), 0, sin(yaw0), cos(yaw0), 0, 0, 0, 1;
  solver_process(N, tf, epochs, x0, R0, xfp, yawf,
                 cyl_o, cyl_r, cyl_h, q,qf,r,yawgain,rpgain,dir_gain,
                 stiffness, stiff_mult, xs);
  //Construct return object
  PyObject* listObj = PyList_New(xs.size());
  if (!listObj) throw logic_error("Failed to make list Object");
  for (int ii = 0; ii < xs.size(); ++ii) {
    Vector3d pos = xs[ii].p;
    Matrix3d rot = xs[ii].R;
    PyObject *tuple = Py_BuildValue("(ddd)((ddd),(ddd),(ddd))",
                                     pos(0),pos(1),pos(2),
                                     rot(0,0),rot(0,1),rot(0,2),
                                     rot(1,0),rot(1,1),rot(1,2),
                                     rot(2,0),rot(2,1),rot(2,2));
    if (!tuple) {
      Py_DECREF(listObj);
      throw logic_error("Failed to make tuple");
    }
    PyList_SET_ITEM(listObj, ii, tuple);
  }
  return listObj;
}

static PyMethodDef pyMethods[] = {
  {"trajgen", gcophrotor_trajgen, METH_VARARGS, "Generate a trajectory."},
  {"trajgen_R", gcophrotor_trajgen_R, METH_VARARGS, "Generate a trajectory (with R fully define)."},
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
