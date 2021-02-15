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



img_hz = 30
queue_time = 2
queue_N = img_hz*queue_time;
queue = [-1] * queue_N
queue_ptr = 0
queue_empty = True

def set_queued_imgs(curr):
        global queue_ptr, queue
	#retVal = (img,queue[(queue_ptr + queue_N/2)%queue_N],queue[queue_ptr])
	queue[queue_ptr] = curr #cat
	queue_ptr = (queue_ptr + 1)%queue_N
	#torch.cat(retVal,1)#change to numpy h/v stack depending on image object type

def get_queued_imgs(seg_image, image, num_imgs=3):
	global queue_empty
	#print("getting")
	#img = image.cpu().detach().numpy().reshape(3,h,w)
	seg_img = seg_image.cpu().detach().numpy()
	#print(image.shape, seg_img.shape)
	curr = np.concatenate((image, seg_img))

	msg = curr.copy()
	#print(msg.shape)

	for i in range(num_imgs-2, -1, -1):
		if queue_empty:
			msg = np.concatenate((msg, curr))

		else:
			loc = (queue_ptr - (img_hz * i))%queue_N
			while queue[loc] is -1:
				loc -= 1
			#print("loc:", loc)
			msg = np.concatenate((msg, queue[loc]))

	set_queued_imgs(curr)
	queue_empty = False

	return msg

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

h = 480
w = 640

stop_thresh = 0.0045
k = 0

#segload = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/model_seg99.pth.tar"
segload = "/home/siddharth/Desktop/asco/ws/src/orange_picking/model/model_seg145.pth.tar"

seg_mean_image = torch.tensor(np.load('data/depth_data/data/mean_seg.npy')).to('cuda')

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
	segmodel = segmodel.to(device)
else:
	device = torch.device('cpu')
	segmodel = segmodel.to(device)

bridge = CvBridge()

pub = rospy.Publisher("/goal_points",PoseArray,queue_size=50)
pub_stop = rospy.Publisher("/stop_node", Bool, queue_size=10)
pub_seg = rospy.Publisher('/orange_picking/seg_image', Image,queue_size=50)

def seg_node_callback(data):
	t_out1 = time.time()
	try:
		image_arr = bridge.imgmsg_to_cv2(data,"passthrough")
	except CvBridgeError as e:
		print(e)

	image_arr = image_arr.transpose(2,0,1)

        image_arr3 = image_arr[::-1, :, :].copy()

	image_arr2 = image_arr3.reshape((1,3,h,w))

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
        cv2.imwrite('test.png', pub_np)

	print(pub_np.shape)
	#pub_seg.publish(bridge.cv2_to_imgmsg(pub_np,"passthrough"))
	print(np.sum(seg_np), w*h, float(np.sum(seg_np))/(float(w) * float(h)))
        segimages = segimages.type(torch.FloatTensor).to(device)
	if np.sum(seg_np) >= (stop_thresh * w * h):
		msg = PoseArray()
		msg_stop = Bool()

		for pt in range(num_pts):
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
		print("STOP STOP STOP")
		#goal = np.array(goal)
		return


	segimages -= seg_mean_image
	#segimages = torch.reshape(segimages, (segimages.shape[0], 1, segimages.shape[1], segimages.shape[2]))
	msg = get_queued_imgs(segimages, image_arr, 3)
	#print(msg.shape)
	msg = np.transpose(msg, (1, 2, 0))
	try:
		print("pub_seg")
		pub_seg.publish(bridge.cv2_to_imgmsg(msg,"passthrough"))
	except CvBridgeError as e:
		print(e)
		exit(0)

	#seg_tensor_image = torch.cat((image_tensor, segimages), 1)


def seg_node():
	rospy.init_node('image_segmentation')
	rospy.Subscriber("/orange_picking/processed_image", Image, seg_node_callback)
	rospy.spin()

if __name__ == "__main__":
	try:
		seg_node()
	except rospy.ROSInterruptException:
		pass
