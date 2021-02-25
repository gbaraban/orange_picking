import rosbag
import numpy as np

def checkStop(msg):
  if msg....
    return True
  return False

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('bag_dir', help='bag dir')
  args = parser.parse_args()
  seg_topic = ""
  odom_topic = ""
  stop_topic = ""
  depth_topic = ""
  stop_thresh = 10
  bag_list = os.listdir(args.bag_dir)
  orange_list = []
  xf_list = []
  for bag_name in bag_list:
    print(bag_name)
    if not bag_name.endswidth(".bag"):
      continue
    filename = args.bag_dir + '/' + bag_name
    bag = rosbag.Bag(filename)
    last_x = None
    last_seg_image = None
    last_depth_image = None
    stop_ctr = 0
    for topic, msg, t in bag.read_messages(topics=[stop_topic,odom_topic,seg_topic,depth_topic]):
      if topic == odom_topic:
        last_x = msg
      if topic == seg_topic:
        last_seg_image = msg
      if topic == depth_topic:
        last_depth_image = msg
      if topic == stop_topic:
        if checkStop(msg):
          if stop_ctr > stop_thresh:
            break
          else:
            stop_ctr += 1
        else:
            if stop_ctr > 0:
              print("Resetting: " + stop_ctr)
            stop_ctr = 0
    if stop_ctr > 0:
      print("Stop Found: " + stop_ctr)
      centroid = getCentroid(last_seg_image)
      local_p = getTransform(last_depth_image,centroid)
      xf_p = np.array((last_x.transform.translation.x,
                       last_x.transform.translation.y,
                       last_x.transform.translation.z))
      xf_R = R.from_quat([
                      last_x.transform.rotation.x, 
                      last_x.transform.rotation.y,
                      last_x.transform.rotation.z, 
                      last_x.transform.rotation.w
                     ]).as_dcm()
      orange_list.append(xf_p + xf_R*local_p)
      xf_list.append((xf_p,xf_R))
    else:
      print("No Stop Found")
  #Average orange_pos
  avg_orange = sum(orange_list)/len(orange_list) 
  print("Average Orange: " + avg_orange)
  success_list = [1 if objectiveFunction(xf,avg_orange) else 0 for xf in xf_list]
  print("Accuracy: " + sum(success_list) + " out of " + len(success_list))

if __name__ == '__main__':
    main()
