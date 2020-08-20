###################################
launch folder guide:
###################################

This folder contains all of the launch files associated with the orange picking project:

trajecory.launch:
Defines all the ROS params used by the trajectory node, launches the trajectory node, then launches static_transform_publisher nodes that make the transforms work with the opitrak and realsense systems.

transforms.launch:
Defines all the ROS params used by the trajectory node, then launches static_transform_publisher nodes that make the transforms work with the opitrak and realsense systems.
