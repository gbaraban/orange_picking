import os
import pickle


locs = [ "data_collection/", "data_collection3/", "data_collection4/"] #, "data_collection2/"] #"depth_data/data/",
time_multiplier = [1.0, 1.0, 1.0, 1.0]

kw = "head_on" #keyword to look for
csv = [ None, None, None] # "depth_data.csv", 

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

dest = "combined_data5"
os.makedirs(dest + "/real_world_traj_bag")
os.makedirs(dest + "/real_world_traj_bag_np")
os.makedirs(dest + "/seg_mask")
os.makedirs(dest + "/seg_mask_np")


ctr = 0
for sel_no, loc in enumerate(locs):
	l = len(os.listdir(loc+"real_world_traj_bag"))
	if selected[sel_no] is None:
		fr = range(l)
	else:
		fr = selected[sel_no]

	for i in fr:
		os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/real_world_traj_bag/"+"bag"+str(i) +  " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/real_world_traj_bag/bag" + str(ctr))
		os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/real_world_traj_bag_np/"+"bag"+str(i) + " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/real_world_traj_bag_np/bag" + str(ctr))
		pkl_data = pickle.load(open("/home/gabe/ws/ros_ws/src/orange_picking/data/"+dest+"/real_world_traj_bag/bag" + str(ctr) + "/data.pickle_no_parse", "rb"))
		pkl_data["time_multiplier"] = time_multiplier[sel_no]
		pickle.dump(pkl_data, open("/home/gabe/ws/ros_ws/src/orange_picking/data/"+dest+"/real_world_traj_bag/bag" + str(ctr) + "/data.pickle_no_parse", "wb"))
		if os.path.isdir("/home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/seg_mask/bag" +str(i)):
			os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/seg_mask/"+"bag"+str(i) + " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/seg_mask/bag" + str(ctr))
			os.system("ln -s /home/gabe/ws/ros_ws/src/orange_picking/data/" + loc+"/seg_mask_np/"+"bag"+str(i) + " /home/gabe/ws/ros_ws/src/orange_picking/data/" + dest + "/seg_mask_np/bag" + str(ctr))

		ctr += 1


