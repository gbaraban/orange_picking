import numpy as np
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('success_dir', type=str, default="", help="location to determine if success")
    parser.add_argument('prefix', type=str, default="data_dagger_Fig30_400_10.0_", help="")
    parser.add_argument('trial_dir', nargs='+', help='bag dir')
    args = parser.parse_args()
    print(args.success_dir, args.trial_dir)
    arm_offset = np.array((1.2,0,0))
    err_list = []
    success_loc = args.success_dir + "/" + args.prefix
    for no, folder in enumerate(args.trial_dir):
        run_metadata = pickle.load(open(args.trial_dir[no]+"/run_metadata.pickle", "rb"))
        head_on = run_metadata["head_on"][0]
        print(head_on)
        trial_list = os.listdir(folder)
        for trial_name in trial_list:
            if trial_name.startswith("run_metadata"):
                continue
            fname = folder + "/" + trial_name
            print(fname)
            tr = int(trial_name.lstrip("trial"))
            #if tr not in head_on:
            #    continue
            if not os.path.isfile(success_loc + trial_name):
                continue
            success_info = open(success_loc + trial_name, "r")
            is_success = False
            for line in success_info:
                l = line.strip("\n").strip().split(" ")
                #print(l[1] + "asdf")
                if l[1] != "None":
                    is_success = True

            if not is_success:
                err_list.append(float('inf'))
                continue
            if os.path.isfile(fname+'/metadata.pickle'):
                with open(fname+'/metadata.pickle','rb') as f:
                    metadata = pickle.load(f)
                    orange = metadata['orangeTrue']
            elif os.path.isfile(fname + "/ref_traj.pickle"):
                print("\nCHECK THIS BAG ", trial_name, "\n") 
                with open(fname+'/ref_traj.pickle','rb') as f:
                    refdata = pickle.load(f)
                    final_state = refdata[len(refdata)-1]
                    orange = np.array(final_state[0]) + R.from_dcm(final_state[1]).apply(arm_offset)
            else:
                print("Neither Metadata nor ref traj found")
                continue
            if os.path.isfile(fname+'/traj_data.pickle'):
                with open(fname+'/traj_data.pickle','rb') as f:
                    trajdata = pickle.load(f)
            else:
                continue
            min_dist = float('inf')
            le = len(trajdata)
            for state in trajdata[-int(0.1*le):]:
                pos = np.array(state[0])
                rot = R.from_dcm(state[1])
                dist = np.linalg.norm(pos + rot.apply(arm_offset) - orange)
                dist = np.abs(np.linalg.norm(pos-orange)-arm_offset[0])
                min_dist = min((min_dist,dist))
            err_list.append(min_dist)
    bin_size = 0.05
    num_bins = int(1.0/bin_size)
    err_bins = [ii*bin_size for ii in range(num_bins)]
    err_bins = np.array(err_bins)
    counts = []
    for ii in range(num_bins):
        counts.append(sum(1.0 for err in err_list if (err < (bin_size + err_bins[ii]))))
    print(counts[-1])
    print(len(err_list))
    fracs = np.zeros(num_bins)
    fracs[0] = 100.0*float(counts[0])/len(err_list)
    for ii in range(1,num_bins):
        fracs[ii] = 100.0*float(counts[ii] - counts[ii-1])/len(err_list)
#    fracs = [(100.0*(counts[ii] - counts[max((ii-1,0))])/len(err_list)) for ii in range(num_bins)]
    print(err_bins)
    print(fracs)
    #for ii in range(1,num_bins):
    #    counts[ii] = counts[ii] - counts[ii-1]
    #counts = 100*np.array(counts)/trials
    fig = plt.figure()
    ax = plt.axes
    plt.bar(err_bins-(bin_size/2),fracs,width=bin_size)
    plt.xlabel("Error")
    plt.ylabel("Percentage")
    plt.show()

if __name__ == '__main__':
    main()
