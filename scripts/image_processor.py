import rospy
from sensor_msgs.msg import Image
import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError
import cv2

mean_image = "~/Desktop/asco/ws/orange_picking/data/mean_imgv2_data_real_world_traj_bag.npy"

if not (os.path.exists(mean_image)):
	print('mean image file not found', mean_image)
	exit(0)
else:
	print('mean image file found')
	mean_image = np.load(mean_image)

mean_image = mean_image.transpose(2,0,1)

bridge = CvBridge()

pub = rospy.Publisher('/orange_picking/processed_image', Image, queue=30)

def processing_callback(data):
	try:
		cv_image = bridge.imgmsg_to_cv2(data, "passthrough")
	except CvBridgeError as e:
		print(e)

	print(cv_image.shape)

	if mean_image is None:
		mean_subtracted = (cv_image)
	else:
		mean_subtracted = (cv_image-mean_image)
	
	#image_tensor = torch.tensor(mean_subtracted)

	try:
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
