"""
#  setting the environment for each nodes (agent)
#  get the log data from apis-emulator for states

@author: Qiong
"""

import logging.config
import time

import numpy as np
import random

logger = logging.getLogger(__name__)

# import global_var as gl
# import config as conf

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


class APIS():
    def __init__(self):
        # self.action_space = ["excess", "sufficient", "scarce", "short"]
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
            # action_accept =

        # minimize purchase from the powerline
        # receiving states: pv , load, p2, rsoc
        # powerline_energy = power_flow_to_battery - load ?
        # reward = powerline_energy
        # reward = p2

        # return next_state, reward

    def step1(self, state, action_request, action_accept):
        current_pv = state[0]
        current_load = state[1]
        current_p2 = state[2]
        current_rsoc = state[3]
        current_rsoc_ave = state[4]

        # return reward

    def reset(self):
        # reset the states according to standard.json file (../apis-emulator/jsontmp)
        # all values are the same to each house
        # super().reset(seed=seed)

        # init state
        pvc_charge_power = np.array([0])
        ups_output_power = np.array([0])
        p2 = np.array([0])
        rsoc = np.array([50])
        wg = 0
        wb = -4.5

        rsoc_ave = np.array([50])  # average rsoc in the same community

        # self.state = np.array([self.state])
        self.state = np.concatenate([pvc_charge_power, ups_output_power, p2, rsoc, rsoc_ave], axis=-1)

        return np.array(self.state, dtype=np.float32)
