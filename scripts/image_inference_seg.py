import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import torch
from orangenetarch import *
import numpy as np
import time, os
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.spatial.transform import Rotation as R
from segmentation.segmentnetarch import *
from sensor_msgs.msg import Image


capacity = 1.0
num_images = 1
num_pts = 3
bins = 30
outputs = 6
resnet18 = False
if torch.cuda.is_available():
	gpu = 0
else:
	gpu = None

#load = "/home/gabe/ws/ros_ws/src/orange_picking/test_run/real_world_data_aug_more_data/modelLast.pth.tar"
load = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/model79.pth.tar"
load = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/10_19_seg/modelLast.pth.tar" #seg retrain
load = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/10_19_seg_off/modelLast.pth.tar" #seg retrain off
load = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/real_world_fixed_aug/model14.pth.tar"
load = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/real_world_fixed_aug_retrain_off/model14.pth.tar"


segload = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/model_seg99.pth.tar"
mins = [(0.0,-0.5,-0.1,-np.pi,-np.pi/2,-np.pi),(0.0,-1,-0.15,-np.pi,-np.pi/2,-np.pi),(0.0,-1.5,-0.2,-np.pi,-np.pi/2,-np.pi),(0.0,-2.0,-0.3,-np.pi,-np.pi/2,-np.pi),(0.0,-3.0,-0.5,-np.pi,-np.pi/2,-np.pi)]
maxs = [(1.0,0.5,0.1,np.pi,np.pi/2,np.pi),(2.0,1.0,0.15,np.pi,np.pi/2,np.pi),(4.0,1.5,0.2,np.pi,np.pi/2,np.pi),(6.0,2.0,0.3,np.pi,np.pi/2,np.pi),(7.0,0.3,0.5,np.pi,np.pi/2,np.pi)]


if not resnet18:
	model = OrangeNet8(capacity,num_images,num_pts,bins,mins,maxs,n_outputs=outputs,num_channels=4)
else:
	model = OrangeNet18(capacity,num_images,num_pts,bins,mins,maxs,n_outputs=outputs,num_channels=4)

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


segmodel = SegmentationNet()

if os.path.isfile(segload):
        if not gpu is None:
                checkpoint = torch.load(segload,map_location=torch.device('cuda'))
        else:
                checkpoint = torch.load(segload)
        segmodel.load_state_dict(checkpoint)
        segmodel.eval()
        print("Loaded Model: ", segload)
else:
        print("No checkpoint found at: ", segload)
        exit(0)


if not gpu is None:
	device = torch.device('cuda:'+str(gpu))
	model = model.to(device)
	segmodel = segmodel.to(device)
else:
	device = torch.device('cpu')
	model = model.to(device)
	segmodel = segmodel.to(device)


bridge = CvBridge()

pub = rospy.Publisher("/goal_points",PoseArray,queue_size=50)
pub_stop = rospy.Publisher("/stop_node", Bool, queue_size=10)
pub_seg = rospy.Publisher('/orange_picking/seg_image', Image,queue_size=50)

h = 480
w = 640

stop_thresh = 0.025
k = 0
mean_image = "/home/gabe/ws/ros_ws/src/orange_picking/test_run/mean_imgv2_data_real_world_traj_bag.npy"
mean_image = "/home/siddharth/Desktop/asco/ws/src/orange_picking/data/mean_imgv2_data_depth_data_data_real_world_traj_bag.npy" #"mean_imgv2_data_real_world_traj_bag480.npy"

seg_mean_image = torch.tensor(np.load('data/depth_data/data/mean_seg.npy')).to('cuda')

if not (os.path.exists(mean_image)):
        print('mean image file not found', mean_image)
        exit(0)
else:
        print('mean image file found')
        mean_image = np.load(mean_image)


def inference_node_callback(data):
	#global k
	t_out1 = time.time()
	try:
		image_arr = bridge.imgmsg_to_cv2(data,"passthrough")
	except CvBridgeError as e:
		print(e)

	#np.save("test/test"+str(k) + ".npy", image_arr)
	#image_arr2 = ((image_arr + mean_image)*255.0).astype(int)
	#cv2.imwrite('test/test' +str(k) + '.png', image_arr2)
	#k += 1
	#image_arr = np.load("data/depth_test/image350.npy")
	image_arr = image_arr.transpose(2,0,1)
	#print(image_arr.shape)
	image_arr2 = image_arr.reshape((1,3,h,w))
	#print(image_arr.shape)
	#if mean_image is None:
	#	mean_subtracted = (image_arr)
	#else:
	#	mean_subtracted = (image_arr-mean_image)
	#print(np.all(image_arr == image_arr2[0,:,:,:]))
	image_tensor = torch.tensor(image_arr2)

	image_tensor = image_tensor.to(device,dtype=torch.float)

	#image_tensor = image_tensor.permute(2,0,1).unsqueeze(0)
	t_in1 = time.time()

	seglogits = segmodel(image_tensor)
	seglogits = seglogits.view(-1,2,segmodel.h,segmodel.w)
	segimages = (torch.max(seglogits, 1).indices).to('cpu') #.type(torch.FloatTensor).to(device)
	#seg_cpu = segimages.to('cpu')
	seg_np = np.array(segimages[0,:,:]) #+ mean_image
	pub_np = (255 * seg_np).astype(np.uint8).reshape((h, w))
	print(pub_np.shape)
	pub_seg.publish(bridge.cv2_to_imgmsg(pub_np,"passthrough"))
	print(np.sum(seg_np), w*h, float(np.sum(seg_np))/(float(w) * float(h)))
        segimages = segimages.type(torch.FloatTensor).to(device)
	if np.sum(seg_np) >= (stop_thresh * w * h):
		msg = PoseArray()
		msg_stop = Bool()

		for pt in range(model.num_points):
                	point = [0, 0, 0, 0, 0, 0]
			point = np.array(point)
			#print(point)
			#goal.append(point)
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

		msg_stop.data = True
		pub.publish(msg)
		pub_stop.publish(msg_stop)
		#goal = np.array(goal)
		return


	segimages -= seg_mean_image
	segimages = torch.reshape(segimages, (segimages.shape[0], 1, segimages.shape[1], segimages.shape[2]))

	seg_tensor_image = torch.cat((image_tensor, segimages), 1)
	#print(segimages.shape, batch_imgs.shape)
	logits = model(seg_tensor_image)

	logits = model(seg_tensor_image)
	logits = logits.cpu()
	logits = logits.view(1,model.outputs,model.num_points,model.bins).detach().numpy()
	t_in2 = time.time()
	predict = np.argmax(logits,axis=3)
	#print("Predict: ", predict)
	goal = []
	msg = PoseArray()
	for pt in range(model.num_points):
		point = []
		for coord in range(model.outputs):
			bin_size = (model.max[pt][coord] - model.min[pt][coord])/float(model.bins)
			point.append(model.min[pt][coord] + bin_size*predict[0,coord,pt])
		point = np.array(point)
		#print(point)
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
	#print(goal)
	#print(time_proc)
	#Publish

def inference_node():
	rospy.init_node('image_inference')
	rospy.Subscriber("/orange_picking/processed_image", Image, inference_node_callback)
	rospy.spin()

if __name__ == "__main__":
	try:
		inference_node()
	except rospy.ROSInterruptException:
		pass
