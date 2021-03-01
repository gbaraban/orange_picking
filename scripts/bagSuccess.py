import rosbag
import numpy as np
from scipy import ndimage
from segmentation.segmentnetarch import *
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
import PIL.Image

def depthnp_from_image(image):
  #Stolen from https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
  dtype_class = np.uint16
  channels = 1
  dtype = np.dtype(dtype_class)
  dtype = dtype.newbyteorder('>' if image.is_bigendian else '<')
  shape = (image.height, image.width, channels)

  data = np.fromstring(image.data, dtype=dtype).reshape(shape)
  data.strides = (
    image.step,
    dtype.itemsize * channels,
    dtype.itemsize
  )
  data = ((data[:,:,0].astype(np.float64)/data.max())*255).astype(np.uint8)
  return data


def np_from_image(image):
  #Stolen from https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
  dtype_class = np.uint8
  channels = 3
  dtype = np.dtype(dtype_class)
  dtype = dtype.newbyteorder('>' if image.is_bigendian else '<')
  shape = (image.height, image.width, channels)

  data = np.fromstring(image.data, dtype=dtype).reshape(shape)
  data.strides = (
    image.step,
    dtype.itemsize * channels,
    dtype.itemsize
  )
  return data


def checkStop(msg, num_points=3):
    eps = 1e-5
    point = np.array([0, 0, 0, 0, 0, 0])
    R_quat = R.from_euler('zyx', point[3:6]).as_quat()	
    
    for pt in range(num_points):
        if (np.abs(msg.poses[pt].position.x - point[0]) < eps) and \
            (np.abs(msg.poses[pt].position.y - point[1]) < eps) and \
            (np.abs(msg.poses[pt].position.z - point[2]) < eps) and \
            (np.abs(msg.poses[pt].orientation.x - R_quat[0]) < eps) and \
            (np.abs(msg.poses[pt].orientation.y - R_quat[1]) < eps) and \
            (np.abs(msg.poses[pt].orientation.z - R_quat[2]) < eps) and \
            (np.abs(msg.poses[pt].orientation.w - R_quat[3]) < eps):
            pass
        else:
            return False

    return True

def getCentroid(img, segmodel, mean_image,device):
    if img is None:
        print("Img is None")
        return None
    h = 480
    w = 640
    img_np = np_from_image(img)
    img_np = img_np/255.0
    image_arr = img_np - mean_image
    image_arr = image_arr.transpose(2,0,1)
    image_arr = image_arr.reshape((1,3,h,w))
    image_tensor = torch.tensor(image_arr)
    image_tensor = image_tensor.to(device,dtype=torch.float)

    seglogits = segmodel(image_tensor)
    seglogits = seglogits.view(-1,2,segmodel.h,segmodel.w)
    segimages = (torch.max(seglogits, 1).indices).to('cpu') #.type(torch.FloatTensor).to(device)
    seg_np = np.array(segimages[0,:,:])
    c = ndimage.measurements.center_of_mass(seg_np)
    if np.isnan(c[0]) or np.isnan(c[1]):
        return None
    #if (c[0] < 100) or (c[1] < 100):
    #    PIL.Image.fromarray((seg_np*255).astype('uint8')).show()
    return c

def objectiveFunction(depth_im, c):
    if depth_im is None:
        print("Depth Img is None")
        return False
    depth_arr = depthnp_from_image(depth_im)
    c_fl = (int(c[0]),int(c[1]))
    frac = (float(c[0])-c_fl[0],float(c[1])-c_fl[1])
    c_ceil = (c_fl[0]+1,c_fl[1]+1)
    ctr_d4 = np.zeros((2,2))
    ctr_d4[0,0] = depth_arr[c_fl[0],c_fl[1]]
    ctr_d4[0,1] = depth_arr[c_fl[0],c_ceil[1]]
    ctr_d4[1,0] = depth_arr[c_ceil[0],c_fl[1]]
    ctr_d4[1,1] = depth_arr[c_ceil[0],c_ceil[1]]
    ctr_d = ctr_d4[0,0] + frac[0]*(ctr_d4[1,0]-ctr_d4[0,0]) + frac[1]*(ctr_d4[0,1]-ctr_d4[0,0])
    if (c[0] < 100) or (c[1] < 100):
        print("U-V failed")
        return False
    if (ctr_d > 29) or (ctr_d < 1):
        return False
    print("Centroid: ",c," Depth: ",ctr_d)
    return True
    #TODO: add check on u,v or do back projection

def processRun(img,depth_im,start_time,end_time,stop_ctr,model, mean_image,device,bag_start_time = None):
    time_thresh = 10
    stop_thresh = 5
    duration = end_time - start_time
    if duration.to_sec() < time_thresh:
        #print(duration.to_sec(),"s too short")
        return (0,0)
    print(duration.to_sec(),"s run found, with ", stop_ctr, " stops")
    if stop_ctr < stop_thresh:
        return (0,1)
    centroid = getCentroid(img, model, mean_image,device)
    if centroid is None:
        return (0,1)
    if (objectiveFunction(depth_im,centroid)):
        if bag_start_time:
            print("Start: ",(start_time-bag_start_time).to_sec()," End: ",(end_time-bag_start_time).to_sec())
        return (1,1)
    return (0,1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_dir', nargs='+', help='bag dir')
    args = parser.parse_args()
    img_topic = "/camera/color/image_raw"
    odom_topic = "/gcop_odom"
    stop_topic = "/goal_points"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    status_topic = "/rqt_gui/system_status"
    stop_thresh = 10
    success = 0
    trials = 0
    segload =  "/home/gabe/ws/ros_ws/src/orange_picking/model/segmentation/logs/variable_log/2021-01-31_13-25-31/model_seg145.pth.tar" 
    if torch.cuda.is_available():
        gpu = 0
    else:
        gpu = None
    model = SegmentationNet()
    if os.path.isfile(segload):
        if not gpu is None:
            checkpoint = torch.load(segload,map_location=torch.device('cuda'))
        else:
            checkpoint = torch.load(segload)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Loaded Model: ", segload)
    else:
        print("No checkpoint found at: ", segload)
        exit(0)
    if not gpu is None:
        device = torch.device('cuda:'+str(gpu))
        model = model.to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
    mean_image = "data/mean_imgv2_data_depth_data_data_real_world_traj_bag.npy"#data/combined_data4/mean_imgv2_data_combined_data4_real_world_traj_bag.npy"
    if not (os.path.exists(mean_image)):
        print('mean image file not found', mean_image)
        exit(0)
    else:
        print('mean image file found')
        mean_image = np.load(mean_image)
    bag_list = []
    for temp in args.bag_dir:
        bag_list = bag_list + os.listdir(temp)
    for bag_name in bag_list:
        print(bag_name)
        if not bag_name.endswith(".bag"):
            continue
        folder_idx = 0
        filename = args.bag_dir[folder_idx] + '/' + bag_name
        while not (os.path.exists(filename)):
            folder_idx += 1
            filename = args.bag_dir[folder_idx] + '/' + bag_name
        bag = rosbag.Bag(filename)
        last_x = None
        last_image = None
        last_depth_image = None
        last_stop_image = None
        last_stop_depth_image = None
        stop_ctr = 0
        status_on_time = None
        last_stop_time = None
        status = False
        temp_t = None
        for topic, msg, t in bag.read_messages(topics=[stop_topic,img_topic,depth_topic,status_topic]):
            if temp_t is None:
                temp_t = t
            if topic == img_topic:
                last_image = msg
            if topic == depth_topic:
                last_depth_image = msg
            if topic == stop_topic:
                if status and checkStop(msg):
                    stop_ctr += 1
                    last_stop_time = t
                    last_stop_image = last_image
                    last_stop_depth_image = last_depth_image
                    if stop_ctr > 10:
                        retval = processRun(last_stop_image,last_stop_depth_image,status_on_time,last_stop_time,stop_ctr,model,mean_image,device,temp_t)
                        success += retval[0]
                        trials += retval[1]
                        status = False
                        last_image = None
                        last_depth_image = None
                        stop_ctr = 0
                else:
                    #if stop_ctr > 0:
                    #    print("Resetting: ",stop_ctr)
                    stop_ctr = 0
            if topic == status_topic:
                if (not status) and ("PathFollow" in msg.data):
                    status = True
                    status_on_time = t
                elif status and ("PathFollow" not in msg.data):
                    if last_stop_time is None:
                        last_stop_time = t
                    retval = processRun(last_stop_image,last_stop_depth_image,status_on_time,last_stop_time,stop_ctr,model,mean_image,device,temp_t)
                    success += retval[0]
                    trials += retval[1]
                    status = False
                    last_image = None
                    last_depth_image = None
                    stop_ctr = 0
        if status:
            retval = processRun(last_stop_image,last_stop_depth_image,status_on_time,last_stop_time,stop_ctr,model,mean_image,device,temp_t)
            success += retval[0]
            trials += retval[1]
    print("Accuracy: ",float(success)/trials, "%. ",success," out of ",trials)

if __name__ == '__main__':
    main()
