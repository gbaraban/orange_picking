###################################
src folder guide:
###################################

This folder contains all of the c++ files associated with the orange picking project:

trajectory_node_cpp.cpp:
This file defines a ROS node that runs GCOP trajectory generation on received goal point lists.

hrotorpython.cc:
This file defines a C++-Python bridge that allows Python scripts to call several GCOP-based functions.
Details on the specific functions can be found in the file itself.

setup.py:
This file is used by Python to compile hrotorpython.cc into a .so file which can be accessed by python scripts.
Details on editing it can be found in the file and in hrotorpython.cc, but to recompile, do the following:
1. Remove the existing build folder.
2. From the src directory, run: python3 setup.py build
3. Find the new .so file (typically in build/lib.linux-x86_64-3.7/...)
4. Copy that .so file into ../scripts
5. Scripts should be able to import gcophrotor and use its function
