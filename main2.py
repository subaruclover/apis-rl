"""
DQN training, single run, house E002

created by: Qiong

"""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import logging.config
logger = logging.getLogger(__name__)

import time
import global_var as gl
import config as conf
import requests, json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# RL_learn functions
"""
class DQNNet : Deep Q-network Model
class Memory : Memory model
class BatteryEnv: my battery model -> replaced with APIS battery model
"""
from RL_learn import DQNNet, Memory, BatteryEnv

# start_time = time.time()

##############################
# Data loading
# get log data for states
host = conf.b_host
port = conf.b_port
# url = "http://0.0.0.0:4390/get/log"
URL = "http://" + host + ":" + str(port) + "/get/log"

# dicts of states for all houses
pvc_charge_power = {}
ups_output_power = {}
p2 = {}  # powermeter.p2, Power consumption to the power storage system [W]
rsoc = {}
wg = {}  # meter.wg, DC Grid power [W]
wb = {}  # meter.wb, Battery Power [W]

pv_list = []
load_list = []
p2_list = []

# need to refresh the output data every 5s? time.sleep()
while not gl.sema:  # True, alter for different time periods
    # # refresh every 5 seconds
    # time.sleep(5)
    # read variables from /get/log url
    # print(output_data.text)
    output_data = requests.get(URL).text
    output_data = json.loads(output_data)  # dict

    for ids, dict_ in output_data.items():  # ids: E001, E002, ... house ID
        # print('the name of the dictionary is ', ids)
        # print('the dictionary is ', dict_)
        # when ids is "E002" (change to other house ID for other houses)
        pvc_charge_power[ids] = output_data[ids]["emu"]["pvc_charge_power"]
        ups_output_power[ids] = output_data[ids]["emu"]["ups_output_power"]
        p2[ids] = output_data[ids]["dcdc"]["powermeter"]["p2"]
        rsoc[ids] = output_data[ids]["emu"]["rsoc"]
        wg[ids] = output_data[ids]["dcdc"]["meter"]["wg"]
        wb[ids] = output_data[ids]["dcdc"]["meter"]["wb"]

        print("pv of {ids} is {pv},".format(ids=ids, pv=pvc_charge_power[ids]),
              "load of {ids} is {load},".format(ids=ids, load=ups_output_power[ids]),
              "p2 of {ids} is {p2},".format(ids=ids, p2=p2[ids]),
              "rsoc of {ids} is {rsoc},".format(ids=ids, rsoc=rsoc[ids]),
              "wg of {ids} is {wg},".format(ids=ids, wg=wg[ids]),
              "wb of {ids} is {wb},".format(ids=ids, wb=wb[ids])
              )

    # refresh every 5 seconds
    # print("\n")
    # time.sleep(5)

    # scenario files
    # interval = 60 * 60  # every 60s
    # command = createJson()
    # run(interval, command)

        # States  pvc_charge_power[ids], for house E002
        if ids == "E002":
            pv_e002 = np.array([pvc_charge_power["E002"]])
            load_e002 = np.array([ups_output_power["E002"]])
            p2_e002 = np.array([p2["E002"]])
            rsoc_e002 = np.array([rsoc["E002"]])

            x_e002 = np.concatenate([pv_e002, load_e002, p2_e002, rsoc_e002], axis=-1)
            print(x_e002)

        state_size = (4, )
        action_feature = 3  # batteryStatus, request, accept
        learning_rate = 0.01

        # Training hyperparameters
        batch_size = 256
        # EPI = 10

        # Exploration hyperparameters for epsilon greedy strategy
        explore_start = 1.0  # exploration probability at start
        explore_stop = 0.01  # minimum exploration probability
        decay_rate = 0.001  # exponential decay rate for exploration prob

        # Q-learning hyperparameters
        gamma = 0.96  # Discounting rate of future reward

        # Memory hyperparameters
        pretrain_length = 10000  # # of experiences stored in Memory during initialization
        memory_size = 10000  # # of experiences Memory can keep

        # battery = BatteryEnv(action_size=action_size)
        # how the battery changes: from APIS
        # action: scenario generation variables (request, accept, etc..)
        # action refresh to create new scenarios

        memory = Memory(memory_size)

        np.random.seed(42)

    time.sleep(5)

