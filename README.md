# apis-rl

## Introduction
The RL is a separate thread of APIS. It reads the states' information for a reinforcement learning framework from 
apis-emulator, and generates refreshing scenario.json files for action policy. Note that each node (house) have its own 
rl learning file and scenario files, as well as apis-main files.

## File description
config.py and global_var.py are files copied from the apis-main. These files are system files.
agent.py is the main file for getting the state variables and generating new scenario.json file. New scenario.json
file will be replaced the older file under the directory apis-main/exe/.


## How to use
Run 

```bash
$ make run-rl
```

Results will be stored under apis-emulator/data/ folder with the names you defined in apis-emulator/config.pay file.
