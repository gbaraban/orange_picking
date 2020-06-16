import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import os
import pickle
import argparse
from scipy.spatial.transform import Rotation as R

def logRot(m):
  m = np.array(m)
  arg = (np.trace(m)-1)/2.0
  phi = 0
  if(arg >= 1):
    phi = 0
  elif(arg <= -1):
    phi = np.pi
  else:
    phi = np.arccos(arg)
  
  sphi = np.sin(phi)

  if (abs(sphi) < 1e-5):
    return np.array([0,0,0])

  temp = (phi/(2.0*sphi))*(m-np.transpose(m))
  return np.array([temp[2,1],temp[0,2],temp[1,0]])

def parse_state(state,targ = None):
    targ = np.array(targ)
    if (len(state) is 2):
        pos = np.array(state[0])
        rot = R.from_matrix(np.array(state[1]))
        ypr = rot.as_euler(seq = 'zyx', degrees = True)
    elif (len(state) is 6):
        pos = np.array(state[0:3])
        ypr = np.array(state[3:6])
        rot = R.from_euler('zyx',ypr)
    if targ is not None:
        df = targ - pos
        df[2] =0
        df = df/np.linalg.norm(df)
        yd = np.arctan2(df[1],df[0])*180/np.pi
        yawcost = 1 - np.dot(df, rot.apply(np.array([1,0,0])))
    else:
        yd = yawcost = None
    logR = logRot(rot.as_matrix())
    return pos[0],pos[1],pos[2], ypr[2], ypr[1], ypr[0], yd, yawcost, logR

def make_step_plot(goals,states,saveFolder=None,name=None):
    #Plot Environment
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #Plot goals 
    goal_x = [pt[0] for pt in goals]
    goal_y = [pt[1] for pt in goals]
    goal_z = [pt[2] for pt in goals]
    ax.plot3D(goal_x,goal_y,goal_z,'g+')
    #Plot states
    for state in states:
        if len(state) is 6:
            p = state[0:3]
            rot_mat = R.from_euler('zyx',state[3:6]).as_matrix()
        elif len(state) is 2:
            p = state[0]
            rot_mat = np.array(state[1])
        colors = ['red','blue','green']
        for ii in [0,1,2]:
            alpha = 0.5
            v = tuple(rot_mat[:,ii])
            ax.plot3D((p[0],p[0] + alpha*v[0]),(p[1],p[1]+alpha*v[1]),(p[2],p[2]+alpha*v[2]),colors[ii])
    if (saveFolder is None) or (name is None):
        plt.show()
    else:
        fig.savefig(saveFolder + 'step_plot' + name + '.png')
        fig.clf()
        plt.close('all')

def make_full_plots(ts,states, targ, cyl_o, cyl_r=0.6, cyl_h=1.6, saveFolder=None, truth=None):
  x_list = []
  y_list = []
  z_list = []
  roll_list = []
  pitch_list = []
  yaw_list = []
  yaw_d_list = []
  yawcost_list = []
  logR0_list = []
  logR1_list = []
  logR2_list = []
  for state in states:
    x,y,z,roll,pitch,yaw,yd,cost, logR = parse_state(state,targ)
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
    roll_list.append(roll)
    pitch_list.append(pitch)
    yaw_list.append(yaw)
    yaw_d_list.append(yd)
    yawcost_list.append(cost)
    logR0_list.append(logR[0])
    logR1_list.append(logR[1])
    logR2_list.append(logR[2])
  #Plot Environment
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  #Plot cylinder
  theta = np.linspace(0, 2*np.pi, 50)
  zs = np.linspace(cyl_o[2], cyl_o[2]+cyl_h, 50)
  thetac, zc = np.meshgrid(theta,zs)
  xc = cyl_r * np.cos(thetac) + cyl_o[0]
  yc = cyl_r * np.sin(thetac) + cyl_o[1]
  ax.plot_surface(xc,yc,zc)
  #Plot x0 and xf
  ax.plot3D((x_list[0],targ[0]),(y_list[0],targ[1]),(z_list[0],targ[2]),'black')
  #Plot trajectory
  ax.plot3D(x_list,y_list,z_list)
  for state in states:
      if len(state) is 2:
          p = state[0]
          rot_mat = np.array(state[1])
      elif len(state) is 6:
          p = state[0:3]
          rot_mat = R.from_euler('zyx',state[3:6]).as_matrix()
      colors = ['red','blue','green']
      for ii in [0,1,2]:
        alpha = 0.5
        v = tuple(rot_mat[:,ii])
        ax.plot3D((p[0],p[0] + alpha*v[0]),(p[1],p[1]+alpha*v[1]),(p[2],p[2]+alpha*v[2]),colors[ii])
  max_range = np.array([max(x_list) - min(x_list),max(y_list) - min(y_list),max(z_list) - min(z_list)]).max()
  mid = np.array([(max(x_list)+min(x_list))/2.0,(max(y_list)+min(y_list))/2.0,(max(z_list)+min(z_list))/2.0])
  ax.set_xlim(mid[0]-max_range/2.0,mid[0]+max_range/2.0)
  ax.set_ylim(mid[1]-max_range/2.0,mid[1]+max_range/2.0)
  ax.set_zlim(max(0,mid[2]-max_range/2.0),mid[2]+max_range/2.0)
  if truth is not None:
      #R0 = R.from_euler('zyx',(yaw_list[0],pitch_list[0],roll_list[0]))
      #p0 = np.array((x_list[0],y_list[0],z_list[0]))
      for point in truth:
          #point_transformed = R0.apply(point[0]) + p0
          #print('Truth: ' + str(point_transformed))
          #ax.plot3D([point_transformed[0]],[point_transformed[1]],[point_transformed[2]],'g+')
          if len(point) is 2:
              pos = point[0]
          elif len(point) is 6:
              pos = point[0:3]
          elif len(point) is 3:
              pos = point
          ax.plot3D([pos[0]],[pos[1]],[pos[2]],'g+')
  #plt.show()
  #Plot x,y,z
  fig1, (ax1,ax2,ax3) = plt.subplots(3,1)
  ax1.plot(ts,x_list)
  ax1.plot(ts[len(ts)-1],targ[0],'g+')
  ax1.set_title('X Position')
  ax2.plot(ts,y_list)
  ax2.plot(ts[len(ts)-1],targ[1],'g+')
  ax2.set_title('Y Position')
  ax3.plot(ts,z_list)
  ax3.plot(ts[len(ts)-1],targ[2],'g+')
  ax3.set_title('Z Position')
  #Plot r,p,y,yd
  fig2, (ax4,ax5,ax6) = plt.subplots(3,1)
  ax4.plot(ts,roll_list)
  ax4.set_title('Roll')
  ax5.plot(ts,pitch_list)
  ax5.set_title('Pitch')
  ax6.plot(ts,yaw_list)
  ax6.set_title('Yaw')
  ax6.plot(ts,yaw_d_list)
  yawf = (180/np.pi)*np.arctan2(cyl_o[1]-targ[1], cyl_o[0]-targ[0])
  ax6.plot(ts[len(ts)-1],yawf,'g+')
  #Plot cost
  fig3, (ax7) = plt.subplots(1,1)
  ax7.plot(ts,yawcost_list)
  ax7.set_title('Yaw Cost')
  #Plot logR
  fig4, (ax8,ax9,ax10) = plt.subplots(3,1)
  ax8.plot(ts,logR0_list)
  ax8.set_title('R0')
  ax9.plot(ts,logR1_list)
  ax9.set_title('R1')
  ax10.plot(ts,logR2_list)
  ax10.set_title('R2')
  #plt.show()
  if saveFolder is None:
    plt.show()
  if saveFolder is not None:
    fig.savefig(saveFolder + 'traj_plot.png')
    fig.clf()
    fig1.savefig(saveFolder + 'posPlot.png')
    fig1.clf()
    fig2.savefig(saveFolder + 'rpyPlot.png')
    fig2.clf()
    fig3.savefig(saveFolder + 'costPlot.png')
    fig3.clf()
    fig4.savefig(saveFolder + 'logRPlot.png')
    fig4.clf()
  plt.close('all')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('fname', help='pickle file')
  args = parser.parse_args()
  with open(args.fname+'/metadata.pickle','rb') as f:
    metadata = pickle.load(f)
  with open(args.fname+'/trajdata.pickle','rb') as f:
    trajdata = pickle.load(f)
  ts = np.linspace(0,metadata['tf'],metadata['N']+1)
  targ = metadata['xf']
  make_full_plots(ts,trajdata,targ,metadata['cyl_o'],metadata['cyl_r'],metadata['h'],None)
