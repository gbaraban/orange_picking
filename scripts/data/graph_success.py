import os

dir = "orange_tracking_data/real_world_traj_bag/"

t_success = 0
t_success_w_reset = 0
t_failure = 0
t = 0
for bag in os.listdir(dir):
	bag_loc = dir + "/" + bag
	for trial in os.listdir(bag_loc):
		trial_loc = bag_loc + "/" + trial
		success = False
		reset = False
		fail = False
		t += 1
		for status in os.listdir(trial_loc):
			#print(status)
			if status.startswith("final"):
				success = True
			elif status.startswith("reset"):
				reset = True
		#print(reset, success)
		if success and reset:
			t_success_w_reset += 1
		elif success:
			t_success += 1
		else:
			print(trial_loc)
			t_failure += 1

print(t_success, t_success_w_reset, t_failure, t)
