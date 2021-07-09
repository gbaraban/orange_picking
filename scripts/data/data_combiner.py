import os
import pickle


locs = [ "data_collection/bagHandParse/", "data_collection3/bagHandParse/", "data_collection4/bagHandParse/", "data_collection2/bagHandParse/", "data_collection2_done/bagHandParse/", "v1_flights/v1/real_world_traj_bag/", "v1_flights/v1_2/real_world_traj_bag/", "v1_flights/v1_3/real_world_traj_bag/", "v1_flights/v1_flights_jun7/real_world_traj_bag/", "v1_flights/v1_flights_jun8/real_world_traj_bag/", "v1_flights/v1_flights_jun8_2/real_world_traj_bag/", "v1_flights/v1_flights_jun8_done/real_world_traj_bag/"] #"depth_data/data/",
locs = ["v1_flights/v1/real_world_traj_bag/", "v1_flights/v1_2/real_world_traj_bag/", "v1_flights/v1_3/real_world_traj_bag/", "v1_flights/v1_flights_jun7/real_world_traj_bag/", "v1_flights/v1_flights_jun8/real_world_traj_bag/", "v1_flights/v1_flights_jun8_2/real_world_traj_bag/", "v1_flights/v1_flights_jun8_done/real_world_traj_bag/", "v1_flights/v1_flights_jun17/real_world_traj_bag/", "v1_flights/weird_bags/real_world_traj_bag/", "v1_flights/v1_flights_jun22/real_world_traj_bag/"] 
#time_multiplier = [1.0, 1.0, 1.0, 1.0]

kw = "head_on" #keyword to look for
csv = [None, None, None, None, None, None, None, None, None, None] # "depth_data.csv", 
#csv = [ None, None, None, None, None, None, None, None, None, None, None, None]

print(len(locs))
print(len(csv))
#exit(0)
selected = []

for i in range(len(locs)):
	if csv[i] is None:
		selected.append(None)

	else:
		f = open(csv[i], "r")
		sel = []
		for lno, line in enumerate(f):
			if lno == 0:
				continue
			else:
				l = line.strip().strip("\n").strip().split(",")
				id, typ = l[0], l[1]
				if typ == kw:
					sel.append(int(l[0]))
		if len(sel) == 0:
			print("Sel len is 0")
		else:
			print(sel)

		selected.append(sel)

dest = "hand_carry_coeff_data"
dest = "orange_tracking_data3"
os.makedirs(dest + "/real_world_traj_bag")
os.makedirs(dest + "/real_world_traj_bag_np")
#os.makedirs(dest + "/seg_mask")
#os.makedirs(dest + "/seg_mask_np")


ctr = 0
for sel_no, loc in enumerate(locs):
	l = len(os.listdir(loc))
	if selected[sel_no] is None:
		fr = range(l)
	else:
		fr = selected[sel_no]

	for i in fr:
		os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"bag"+str(i) +  " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/real_world_traj_bag/bag" + str(ctr))
		#os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"bag"+str(i) + " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/real_world_traj_bag_np/bag" + str(ctr))
		#pkl_data = pickle.load(open("/home/gabe/ws/ros_ws/src/orange_picking/data/"+dest+"/real_world_traj_bag/bag" + str(ctr) + "/data.pickle_no_parse", "rb"))
		#pkl_data["time_multiplier"] = time_multiplier[sel_no]
		#pickle.dump(pkl_data, open("/home/gabe/ws/ros_ws/src/orange_picking/data/"+dest+"/real_world_traj_bag/bag" + str(ctr) + "/data.pickle_no_parse", "wb"))
		if os.path.isdir("/home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/seg_mask/bag" +str(i)):
			os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/seg_mask/"+"bag"+str(i) + " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/seg_mask/bag" + str(ctr))
			os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/seg_mask_np/"+"bag"+str(i) + " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/seg_mask_np/bag" + str(ctr))

		ctr += 1


