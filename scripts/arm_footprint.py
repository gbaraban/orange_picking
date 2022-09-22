#!/usr/bin/env python
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

#from baseline_image_inference import *

class FootprintMaker:
    def __init__(self):
        self.image_topic="/camera/color/image_raw"
        self.depth_topic="/camera/aligned_depth_to_color/image_raw"
        self.bridge = CvBridge()
        self.debug_image_publisher = rospy.Publisher("/debug_image", Image, queue_size=5)
        self.footprint = None
        self.im_sub = rospy.Subscriber(self.image_topic, Image, self.callback)

    def rosmsg2np(self,data):
        try:
            image_arr = self.bridge.imgmsg_to_cv2(data,"rgb8")
        except CvBridgeError as e:
            print(e)
        return image_arr

    def makeFootprint(self,im):
        h = 480
        w = 640
        ret = np.zeros((h,w)).astype(np.uint8)
        arm_bound_x = (250,480)
        arm_bound_y = (250,475)
        #arm_color = [np.array((12,154,226)),np.array((255,255,255))]
        #color_thresh = [90,70]
        bkg_color = np.array((95,150,150))
#        color_thresh = [100]
        for ii in range(arm_bound_x[0],arm_bound_x[1]):
            for jj in range(arm_bound_y[0],arm_bound_y[1]):
                color = im[ii,jj]
                if (color[0] < bkg_color[0]) or (color[1] > bkg_color[1]) or (color[2] > bkg_color[2]):
                    ret[ii,jj] = 255
                #for c,t in zip(bkg_color,color_thresh):
                #    diff = np.linalg.norm(c - color)
                #    if diff < t:
                #        #print(color)
                #        ret[ii,jj] = 255
                #        break
        image_message = self.bridge.cv2_to_imgmsg(ret, encoding="passthrough")
        print("Publishing")
        self.debug_image_publisher.publish(image_message)
        return ret

    def callback(self,image):
        print("callback")
        im_np = self.rosmsg2np(image)
        self.footprint = self.makeFootprint(im_np)

def main():
    rospy.init_node('arm_footprint')
    f = FootprintMaker()
    rospy.spin()
    np.save("footprint.npy",f.footprint)
    return

if __name__ == "__main__":
    main()