import os
import pickle
import shutil


if __name__ == "__main__":
	fr = "seg_mask"
	to = "seg_mask_real"
	#lambda_dict = pickle.load(open("bag_order_lambda.pickle", "rb"))
	#sid_dict = pickle.load(open("bag_order_sid.pickle", "rb"))
	match_dict = pickle.load(open("match.pkl", "rb"))

	for file in os.listdir(fr):
		#fnum = int(file.strip("bag"))
		#fnum2 = lambda_dict[sid_dict[fnum]]

		shutil.copytree(fr+"/"+file, to+"/"+match_dict[file])
