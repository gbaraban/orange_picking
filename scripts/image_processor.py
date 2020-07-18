import rospy
from sensor_msgs.msg import Image
import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError
import cv2

mean_image = "/home/siddharth/Desktop/asco/ws/src/orange_picking/data/mean_imgv2_data_real_world_traj_bag.npy"

if not (os.path.exists(mean_image)):
	print('mean image file not found', mean_image)
	exit(0)
else:
	print('mean image file found')
	mean_image = np.load(mean_image)

#mean_image = mean_image.transpose(0,1,2)

bridge = CvBridge()

pub = rospy.Publisher('/orange_picking/processed_image', Image,queue_size=50)

h = 380
w = 640

def processing_callback(data):
	try:
		cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
		print(e)

	#print(cv_image.shape)
	cv_image = cv2.resize(cv_image,(w,h))
	#cv2.imwrite("test_pre.png", cv_image)
	cv_image = cv_image/255.0
	#print(cv_image.shape)
	if mean_image is None:
		mean_subtracted = (cv_image)
	else:
		mean_subtracted = (cv_image-mean_image)
	#print(mean_subtracted)
	#print(mean_subtracted.shape)
	#image_tensor = torch.tensor(mean_subtracted)

	try:
		print("pub")
		pub.publish(bridge.cv2_to_imgmsg(mean_subtracted,"passthrough"))
	except CvBridgeError as e:
		print(e)


def image_processor():
	rospy.init_node('image_processor')
	rospy.Subscriber("/camera/color/image_raw", Image, processing_callback)
	rospy.spin()


if __name__ == "__main__":
	try:
		image_processor()
	except rospy.ROSInterruptException:
		pass
