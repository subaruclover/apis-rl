"""
#  setting the environment for each nodes (agent)
#  get the log data from apis-emulator for states

@author: Qiong
"""

import logging.config
import time

import numpy as np
import random
import gym
from gym.utils import seeding

logger = logging.getLogger(__name__)

import global_var as gl
import config as conf
import analyser
import core
import requests, json

# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from createScenario import CreateScenario

from RL_learn import DQNNet, SumTree, Memory

"""
# get log data for states
host = conf.b_host
port = conf.b_port
# url = "http://0.0.0.0:4390/get/log"

URL = "http://" + host + ":" + str(port) + "/get/log"

import requests, json
# response = requests.request("POST", url, data=gl)
# print(response.text)
# dicts of states for all houses
pvc_charge_power = {}
ups_output_power = {}
p2 = {}  # powermeter.p2, Power consumption to the power storage system [W]
wg = {}  # meter.wg, DC Grid power [W]
wb = {}  # meter.wb, Battery Power [W]

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
        pvc_charge_power[ids] = output_data[ids]["emu"]["pvc_charge_power"]
        ups_output_power[ids] = output_data[ids]["emu"]["ups_output_power"]
        p2[ids] = output_data[ids]["dcdc"]["powermeter"]["p2"]
        wg[ids] = output_data[ids]["dcdc"]["meter"]["wg"]
        wb[ids] = output_data[ids]["dcdc"]["meter"]["wb"]

        # print("pv of {ids} is {pv},".format(ids=ids, pv=pvc_charge_power[ids]),
        #       "load of {ids} is {load},".format(ids=ids, load=ups_output_power[ids]),
        #       "p2 of {ids} is {p2},".format(ids=ids, p2=p2[ids]),
        #       "wg of {ids} is {wg},".format(ids=ids, wg=wg[ids]),
        #       "wb of {ids} is {wb},".format(ids=ids, wb=wb[ids])
        #       )

    # refresh every 5 seconds
    # print("\n")
    time.sleep(5)

    # scenario files
    # interval = 60 * 60  # every 60s
    # command = createJson()
    # run(interval, command)
"""
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

class APIS():
    def __init__(self):
        # request and accept level: between [0, 1]
        self.action_request_space = np.linspace(0.2, 0.9, 8).tolist()  # [0.2~0.9]
        self.action_accept_space = np.linspace(0.2, 0.9, 8).tolist()  # [0.2~0.9]

        # self.n_actions = len(self.action_request_space) + len(self.action_accept_space)

    """
    # list of possible actions
    # reward
    def step(self, state, action_request, action_accept):

        # Exploration hyperparameters for epsilon greedy strategy
        explore_start = 1.0  # exploration probability at start
        explore_stop = 0.01  # minimum exploration probability
        decay_rate = 0.001  # exponential decay rate for exploration prob
        decay_step = 0  # Decay rate for ϵ-greedy policy

        # action selection
        # ϵ-greedy policy

        # action_request = sorted(np.random.choice(action_request_num, 2, replace=False), reverse=True)  # 2 values
        # action_accept = np.random.choice(action_accept_num, 1, replace=False)

        exp_exp_tradeoff = np.random.rand()
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
            -decay_rate * decay_step
        )

        if explore_probability > exp_exp_tradeoff:
            action_request = np.random.choice()  # 2 values
            action_accept = np.random.choice()  # 1 value
        else:
            action_req = np.argmax(DQN.model.predict(np.expand_dims(state, axis=0)))

        # minimize purchase from the powerline
        # receiving states: pv , load, p2, rsoc
        # powerline_energy = power_flow_to_battery - load ?
        # reward = powerline_energy
        # reward = p2

        return next_state, reward
        # return reward

    # def reset(self):

    """

    def CreateSce(self, action_request, action_accept):
        newSce = CreateScenario(action_request=action_request, action_accept=action_accept)
        newSce.write_json()

        # if __name__ == "__main__":
        #     interval = 60 * 60  # every 60 * 60s
        #     command = createJson()
        #     run(interval, command)


# House Model, step function (reward)

class House():

    def __init__(self):

        self.action_request_space = np.linspace(0.2, 0.9, 8).tolist()
        self.action_accept_space = np.linspace(0.2, 0.9, 8).tolist()

        # list of possible actions
        # reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, state, action_request, action_accept):

        # Exploration hyperparameters for epsilon greedy strategy
        explore_start = 1.0  # exploration probability at start
        explore_stop = 0.01  # minimum exploration probability
        decay_rate = 0.001  # exponential decay rate for exploration prob
        decay_step = 0  # Decay rate for ϵ-greedy policy

        # action selection
        # ϵ-greedy policy

        # action_request = sorted(np.random.choice(action_request_num, 2, replace=False), reverse=True)  # 2 values
        # action_accept = np.random.choice(action_accept_num, 1, replace=False)

        exp_exp_tradeoff = np.random.rand()
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(
            -decay_rate * decay_step
        )

        if explore_probability > exp_exp_tradeoff:
            action_request_num = len(self.action_request_space)
            action_accept_num = len(self.action_accept_space)
            learning_rate = 0.01
            action_request = sorted(np.random.choice(action_request_num, 2, replace=False), reverse=True)  # 2 values
            action_accept = np.random.randint(action_request[1], action_request[0],
                                              1)  # 1 value between 2 request actions
            # action_request = np.random.choice()  # 2 values
            # action_accept = np.random.choice()  # 1 value
        else:
            action_request = np.argmax(DQNNet.model.predict(np.expand_dims(state, axis=0)))
            # np.argmax -> np.argsort, get top 3 indices
            # action_accept =

        # minimize purchase from the powerline
        # receiving states: pv , load, p2, rsoc
        # powerline_energy = power_flow_to_battery - load ?
        # reward = powerline_energy
        # reward = p2

        # return next_state, reward

    def step1(self, state, action_request, action_accept):
        # current_pv = state[0]
        # current_load = state[1]
        # current_p2 = state[2]
        # current_rsoc = state[3]
        # current_rsoc_ave = state[4]

        current_pvc, current_load, current_p2, current_rsoc, current_rsoc_ave = self.state
        # current_pvc = gl.oesunits[ids]["emu"]["pvc_charge_power"]
        # current_rsoc = core.rsocUpdate()

        output_data = requests.get(URL).text
        output_data = json.loads(output_data)  # dict

        rsoc_list = []

        for ids, dict_ in output_data.items():  # ids: E001, E002, ... house ID
            # print('the name of the dictionary is ', ids)
            # print('the dictionary is ', dict_)
            # when ids is "E001" (change to other house ID for other houses)
            pvc_charge_power[ids] = output_data[ids]["emu"]["pvc_charge_power"]
            ups_output_power[ids] = output_data[ids]["emu"]["ups_output_power"]
            p2[ids] = output_data[ids]["dcdc"]["powermeter"]["p2"]
            rsoc[ids] = output_data[ids]["emu"]["rsoc"]
            wg[ids] = output_data[ids]["dcdc"]["meter"]["wg"]
            wb[ids] = output_data[ids]["dcdc"]["meter"]["wb"]

            print("pv of {ids} is {pv},".format(ids=ids, pv=pvc_charge_power[ids]),
                  "load of {ids} is {load},".format(ids=ids, load=ups_output_power[ids]),
                  "p2 of {ids} is {p2},".format(ids=ids, p2=p2[ids]),
                  "rsoc of {ids} is {rsoc},".format(ids=ids, rsoc=rsoc[ids])
                  # "wg of {ids} is {wg},".format(ids=ids, wg=wg[ids]),
                  # "wb of {ids} is {wb},".format(ids=ids, wb=wb[ids])
                  )
            rsoc_list.append(rsoc[ids])
            # refresh every 5 seconds
            # print("\n")
            # time.sleep(5)

            # States  pvc_charge_power[ids], for house E001
            if ids == "E001":
                current_pvc_e001 = np.array([pvc_charge_power["E001"]])
                current_load_e001 = np.array([ups_output_power["E001"]])
                current_p2_e001 = np.array([p2["E001"]])
                current_rsoc_e001 = np.array([rsoc["E001"]])

                current_all_e001 = np.concatenate([current_pvc_e001,
                                                   current_load_e001,
                                                   current_p2_e001,
                                                   current_rsoc_e001], axis=-1)
                print(current_all_e001)  # [39.14 575.58 734.    29.98] E001


        # print(rsoc)
        # {'E001': 29.98, 'E002': 29.99, 'E003': 29.98, 'E004': 29.99}
        rsoc_ave = np.mean(rsoc_list)  # get average rsoc of this community
        # print(rsoc_ave)
        self.state = np.concatenate((current_all_e001, rsoc_ave), axis=-1)

        reward = current_p2_e001
        # done  # time, e.g., one hour



        return reward  #, done

    def reset(self):
        # reset the states according to standard.json file (../apis-emulator/jsontmp)
        # all values are the same to each house
        # super().reset(seed=seed)

        # init state
        pvc_charge_power = np.array([0])
        ups_output_power = np.array([0])
        p2 = np.array([0])
        rsoc = np.array([50])
        # wg = np.array([0])
        # wb = np.array([-4.5])
        rsoc_ave = np.array([50])  # average rsoc in the same community

        # self.state = np.array([self.state])
        self.state = np.concatenate([pvc_charge_power, ups_output_power, p2, rsoc, rsoc_ave], axis=-1)

        # return np.array(self.state, dtype=np.float32)
        return np.array(self.state)
