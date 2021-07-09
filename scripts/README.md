Folder:

architecture: tracking different architecture version, used with trainorangenet_orientation_segmentation.py with arg --arch

autodagger.py and autosimulation.py: run simulation and the dagger for simulation using multiple config combinations

bag*Parse.py: parsing collected data for success or data pre-processing

baseline_image_inference.py and states_image_inference.py: run the real-world model on xavier, states_image_inference.py takes care of most of it

bounding_box.py: find waypoint bounds and other insights on data

visual_servoing_pose_estimation.py: library to find pose of orange

config_writer.py: read and write arguments for a specific to be trained/already trained model

custom*Dataset*.py: pytorch dataloader for different types of data

customTransforms.py: for data manipulation and augmentation in pytorch

poly_wp_node.py: rosnode that does polynomial traj/wp finding, given orange_location

dagger_node.py: to use dagger with both baseline and orangenet together

datacollection.py: collect data from simulation, a lot of data!!

envtest.py: testing simulation

gcopTest.py: way to verify basic gcop works

generate_gifs.py: generate a .gif given bunch of images

model_figs.py: generating figures for simulation

orangedagger.py: running dagger in simulation

orangesimulation.py and orangeimages.py: main code for simulation

orangenet*.py: different orangenet architectures which are generally used, old ones can be put in architecture folder

simSuccess.py and score_*.py: generating metrics for simulation

testorangenet_orientation.py: To test accuracy and other metrics on test dataset, not completely obselete, but needs to be integrated in trainorangenet_orientation_segmentation.py

trainorangenet_orientation_segmentation.py: main training script for orangenet for real-world and simulation
