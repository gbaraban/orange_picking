import rospy
from geometry_msgs.msg import Pose, PoseArray, Point32
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

class tf_class:
    def __init__(self,pos, Rot):
        self.p = pos
        self.Rot = Rot
    
    def __mul__(self.other):
        temp_R = self.Rot * other.Rot
        temp_p = self.Rot.apply(other.p) + self.p
        return tf_class(temp_p,temp_R)

    def inv(self):
        inv_R = self.Rot.inv()
        inv_p = -inv_R.apply(self.p)
        return tf_class(inv_p,inv_R) 

    def posyaw(self):
        return np.hstack((self.p,self.Rot.as_euler('zyx')[0]))

def coeffMatrix(start_py, goal_py, tf):
    error_py = goal_py - start_py
    deg = 9
    dim = 4
    constraints = np.zeros((deg+1,dim))
    basis = np.zeros((deg+1,deg+1))
    constraints[0,0] = error_py[0]
    constraints[0,1] = error_py[1]
    constraints[0,2] = error_py[2]
    constraints[0,3] = error_py[3]
    basis[0:5,:] = basisMatrix(tf)
    basis[5:,:] = basisMatrix(0)
    coeff = np.linalg.solve(basis,constraints)
    return coeff

def basisMatrix(time):
    deg = 9
    dim = 4
    basis = np.zeros((dim+1,deg+1))
    coeff = np.ones(deg+1)
    t_exp = np.zeros(deg+1)
    t_exp[0] = 1
    for ii in range(1,deg):
        t_exp[ii] = t_exp[ii-1]*time
    for row in range(dim+1):
        for col in range(deg+1):
            col_row_diff = col - row
            if (row >= 1) and (col_row_diff >= 0):
                coeff[col] = coeff[col]*(col_row_diff + 1)
            if (col_row_diff >= 0):
                basis[row,col] = t_exp[col_row_diff]*coeff[col]
            else:
                basis[row,col] = 0
    return basis

def coeffToWP(coeff,x0,times):
    wp_list = []
    for time in times:
        basis = basisMatrix(time)
        out = np.matmul(basis,coeff)
        posyaw = x0 + out[0,:]
        velyawrate = out[1,:]
        wp_list.append((posyaw,velyawrate))
    return wp_list

class PolyTraj:
    def __init__(self):
        self.tracker_topic = "/ros_tracker"
        self.wp_topic = "/baseline_wp"
        self.gcop_topic = ""
        self.tracker_sub = Subscriber(self.tracker_topic,PoseArray,self.callback)
        self.odom_sub = Subscriber(self.gcop_topic,Odometry,self.odom_callback)
        self.wp_pub = Publisher(self.wp_topic,PoseArray)
        self.odom = None
        self.start_odom = None
        self.start_time = None
        self.stage_flag = True
        self.camera_transform = tf_class(np.array([xx,yy,zz]),R.from_quat([xx,yy,zz,ww]))
        self.tracking_offset_transform = tf_class(np.array([xx,yy,zz]),R.from_quat([xx,yy,zz,ww]))
        self.staging_tf = tf_class(np.array([0.95,-0.05,0.15]),R.from_euler('zyx',(np.pi,0,0)))
        self.final_tf = tf_class(np.array([0.60,-0.05,0.3]),R.from_euler('zyx',(np.pi,0,0)))
        offset_p = np.array([0.60,-0.05,0.3])
        pred_dt = 0.5
        num_pts = 3
    
    def odom_callback(data):
        pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        Rot = R.from_quat([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])
        self.odom = tf_class(pos,Rot)
        if self.start_posyaw is None:
            self.start_posyaw = self.odom.posyaw()
            self.start_time = data.header.stamp

    def callback(tracker_data):
        if self.odom is None:
            return
        tracking_pos = np.array([tracker_data.poses[0].position.x,tracker_data.poses[0].position.y,tracker_data.poses[0].position.z])
        tracking_Rot = R.from_quat([tracker_data.poses[0].orientation.x,tracker_data.poses[0].orientation.y,tracker_data.poses[0].orientation.z,tracker_data.poses[0].orientation.w])
        track_tf = tf_class(tracking_pos,tracking_Rot)
        track_global = self.odom * self.camera_transform * track_tf * self.tracking_offset_transform
        quad_o_frame = self.tracking_offset_transform.inv() * track_tf.inv() * self.camera_transform()
        #TODO: change this to better reflect state machine/timeout
        if self.stage_flag:
            offset_tf = self.staging_tf
            pos_err = quad_o_frame.p - offset_tf.p
            if (pos_err[0] < 0.05) and (np.linalg.norm(pos_err[1:3]) < 0.05):
                self.stage_flag = False 
                offset_tf = self.final_tf
        else:
            offset_tf = self.final_tf
            pos_err = quad_o_frame.p - offset_tf.p
            if (np.linalg.norm(pos_err) > 1):
                self.stage_flag = True
                offset_tf = self.staging_tf
        goal_tf = track_global*offset_tf
    	max_v = 0.15
    	min_tf = 4.0
    	distance = np.linalg.norm(goal_tf.p - track_global.p)
    	tf = max(distance/max_v,min_tf)
        coeff = coeffMatrix(self.start_posyaw,goal_tf.posyaw(),tf)
        t_duration = (tracker_data.header.stamp - self.start_time)
        t = t_duration.secs() + t_duration.nsecs()/1e9
        wp_t = [t+(ii+1)*self.pred_dt for ii in range(self.num_pts)]
        wp_t = [min(tf,temp) for temp in wp_t]
        wp = coeffToWP(coeff,self.start_posyaw,wp_t)
        msg = PoseArray()
        tf0 = self.odom
        for py in wp:
            p_i = py[0:3]
            R_i = R.from_euler('zyx',(py[3],0,0))
            tf_i = tf0.inv()*tf_class(p_i,R_i)
            pose_i = Pose()
            pose_i.position.x = tf_i.p[0]
            pose_i.position.y = tf_i.p[1]
            pose_i.position.z = tf_i.p[2]
            quat_i = tf_i.R.as_quat()
            pose_i.orientation.x = quat_i[0]
            pose_i.orientation.y = quat_i[1]
            pose_i.orientation.z = quat_i[2]
            pose_i.orientation.w = quat_i[3]
            msg.poses.append(pose_i)
            if self.relative:
                tf_0 = tf_class(p_i,R_i)
        self.wp_pub.publish(msg)
        

def main():
    rospy.init_node('poly_wp')
    pt = PolyTraj()
    rospy.spin()
     
if __name__ = "__main__":
    main()
