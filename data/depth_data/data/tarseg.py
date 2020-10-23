import os
import tarfile

loc = "seg_mask_real"
tloc = "real_world_traj_bag"

tar = tarfile.open("real_world.tar", 'w:gz')
for bag in os.listdir(loc):
	tar.add(tloc + "/" + bag)

tar.close()
