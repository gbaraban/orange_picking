import os
import subprocess

num_procs = 16
data_loc = "real_world_traj_bag"
dirs = os.listdir(data_loc)

#print(dirs)

proc_dir = []
n_dirs = int(len(dirs)/num_procs)
size = 0
for i in range(num_procs):
	proc_dir.append(dirs[i*n_dirs:(i+1)*n_dirs])
	size += len(proc_dir[-1])

if len(dirs) > size:
	old_size = size
	size -= len(proc_dir[-1])
	proc_dir[-1].extend(dirs[old_size:])
	size += len(proc_dir[-1])

print(len(dirs))
print(size)

pr = ""
for i, dir in enumerate(proc_dir):
	procs = " python3 preprocess_data_multi_proc.py "
	for d in dir:
		procs += (d + " ")
	if i+1 != len(proc_dir):
		procs += " | "

	pr += procs

os.system(pr)
	#print(procs)

