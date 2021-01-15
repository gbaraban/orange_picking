import rospy
from sensor_msgs.msg import Image
import torch
from architecture.orangenetarch3 import *
import numpy as np
import time, os
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.spatial.transform import Rotation as R

capacity = 1.0
num_images = 3
num_pts = 3
bins = 30
outputs = 6
resnet18 = False
if torch.cuda.is_available():
	gpu = 0
else:
	gpu = None

load = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/real_world_plain_more_data/modelLast.pth.tar"
#load = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/real_world_data_aug_more_data/modelLast.pth.tar"

mins = [(0.0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0.0,-1.0,-0.15,-np.pi,-np.pi/2,-np.pi),(0.0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0.0,-2.0,-0.3,-np.pi,-np.pi/2,-np.pi),(0.0,-3.0,-0.5,-np.pi,-np.pi/2,-np.pi)]
maxs = [(1.0,0.5,0.1,np.pi,np.pi/2,np.pi),(2.0,1.0,0.15,np.pi,np.pi/2,np.pi),(4.0,1.5,0.2,np.pi,np.pi/2,np.pi),(6.0,2.0,0.3,np.pi,np.pi/2,np.pi),(7.0,0.3,0.5,np.pi,np.pi/2,np.pi)]


if not resnet18:
	model = OrangeNet8(capacity,num_images,num_pts,bins,mins,maxs,n_outputs=outputs)
else:
	model = OrangeNet18(capacity,num_images,num_pts,bins,mins,maxs,n_outputs=outputs)

model.min = mins
model.max = maxs

if os.path.isfile(load):
	if not gpu is None:
		checkpoint = torch.load(load,map_location=torch.device('cuda'))
	else:
		checkpoint = torch.load(load)
	model.load_state_dict(checkpoint)
	model.eval()
	print("Loaded Model: ",load)
else:
	print("No checkpoint found at: ", load)
	exit(0)


if not gpu is None:
	device = torch.device('cuda:'+str(gpu))
	model = model.to(device)
else:
	device = torch.device('cpu')
	model = model.to(device)

bridge = CvBridge()

pub = rospy.Publisher("/goal_points",PoseArray,queue_size=50)


h = 380
w = 640

mean_image = "/home/gabe/ws/ros_ws/src/orange_picking/test_run/mean_imgv2_data_real_world_traj_bag.npy"

if not (os.path.exists(mean_image)):
        print('mean image file not found', mean_image)
        exit(0)
else:
        print('mean image file found')
        mean_image = np.load(mean_image)


def inference_node_callback(data):
	t_out1 = time.time()
	try:
		image_arr = bridge.imgmsg_to_cv2(data,"passthrough")
	except CvBridgeError as e:
		print(e)

	#image_arr2 = ((image_arr + mean_image)*255.0).astype(int)
	#cv2.imwrite('test.png', image_arr2)

	#image_arr = image_arr.transpose(2,0,1)
	#print(image_arr.shape)
	image_arr = image_arr.reshape((1,12,h,w))
	print(image_arr.shape)
	#if mean_image is None:
	#	mean_subtracted = (image_arr)
	#else:
	#	mean_subtracted = (image_arr-mean_image)
	image_tensor = torch.tensor(image_arr)

	image_tensor = image_tensor.to(device,dtype=torch.float)

	#image_tensor = image_tensor.permute(2,0,1).unsqueeze(0)
	t_in1 = time.time()
	logits = model(image_tensor)
	logits = logits.cpu()
	logits = logits.view(1,model.outputs,model.num_points,model.bins).detach().numpy()
	t_in2 = time.time()
	predict = np.argmax(logits,axis=3)
	print("Predict: ", predict)
	goal = []
	msg = PoseArray()
	for pt in range(model.num_points):
		point = []
		for coord in range(model.outputs):
			bin_size = (model.max[pt][coord] - model.min[pt][coord])/float(model.bins)
			point.append(model.min[pt][coord] + bin_size*predict[0,coord,pt])
		point = np.array(point)
		print(point)
		goal.append(point)
		pt_pose = Pose()
		pt_pose.position.x = point[0]
		pt_pose.position.y = point[1]
		pt_pose.position.z = point[2]
		R_quat = R.from_euler('zyx', point[3:6]).as_quat()
		pt_pose.orientation.x = R_quat[0]
		pt_pose.orientation.y = R_quat[1]
		pt_pose.orientation.z = R_quat[2]
		pt_pose.orientation.w = R_quat[3]
		msg.poses.append(pt_pose)
	#exit()
	pub.publish(msg)
	goal = np.array(goal)
	t_out2 = time.time()
	time_proc = t_out2 - t_out1
	time_infer = t_in2 - t_in1
	print(goal)
	print(time_proc)
	#Publish

def inference_node():
	rospy.init_node('image_inference')
	rospy.Subscriber("/orange_picking/seg_image", Image, inference_node_callback)
	rospy.spin()

if __name__ == "__main__":
	try:
		inference_node()
	except rospy.ROSInterruptException:
		pass
