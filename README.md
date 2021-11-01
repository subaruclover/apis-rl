# apis-rl

## Introduction
The RL is a separate thread of APIS. It reads the states' information for a reinforcement learning framework from 
apis-emulator, and generates refreshing scenario.json files for action policy. Note that each node (house) have its own 
rl learning file and scenario files, as well as apis-main files.

## File description
config.py and global_var.py are files copied from the apis-main. These files are system files.
agent.py is the main file for getting the state variables and generating new scenario.json file. New scenario.json
file will be replaced the older file under the directory apis-main/exe/.

## TODO
makefile for apis-rl under the main APIS (makefile)

### consider: 
- venv for make file
- requirement.txt for the dependency of the project
(pandas, tf, keras, numpy, matplotlib, ...)



