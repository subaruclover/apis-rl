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

from createScenario import CreateScenario1, CreateScenario2, CreateScenario3, CreateScenario4

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

    def CreateSce1(self, action_request, action_accept):
        # Create Scenario for house 1 (E001)
        newSce = CreateScenario1(action_request=action_request, action_accept=action_accept)
        newSce.write_json()

    def CreateSce2(self, action_request, action_accept):
        # Create Scenario for house 1 (E001)
        newSce = CreateScenario2(action_request=action_request, action_accept=action_accept)
        newSce.write_json()

    def CreateSce3(self, action_request, action_accept):
        # Create Scenario for house 1 (E001)
        newSce = CreateScenario3(action_request=action_request, action_accept=action_accept)
        newSce.write_json()

    def CreateSce4(self, action_request, action_accept):
        # Create Scenario for house 1 (E001)
        newSce = CreateScenario4(action_request=action_request, action_accept=action_accept)
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

    def step1(self, action_request, action_accept, house_id):
        # TODO
        # each house learn seprate,/ take as one
        # how actions changes the states?
        """
        Perform one step in the environment following the action.
        actions = np.argsort(-actions_value)[:3] e.g. [7, 5, 2]

            @param action_request: [actions[0], actions[2]]
                   action_accept: [actions[1]]
             where actions[0]  "4320.0-": "excess",
                   actions[1]  "-2880.0": "short",
                   actions[2]  "3360.0-4320.0": "sufficient",
                               "2880.0-3360.0": "scarce",

                   ids: house id, string

            @return: (for one house) (pvc_, load_, p2_, rsoc_, rsoc_ave_), reward, done
             where reward is set to p2?
             but when the goal is reached (time up), done is True
        """

        # current_pv = state[0]
        # current_load = state[1]
        # current_p2 = state[2]
        # current_rsoc = state[3]
        # current_rsoc_ave = state[4]

        pvc_e001, load_e001, p2_e001, rsoc_e001, rsoc_ave = self.state
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

            rsoc_list.append(rsoc[ids])
            # refresh every 60 seconds
            # print("\n")
            # time.sleep(60)  # <-- wait for time pass and renew the actions?

            # actions! --> change states
            # action_request: [actions[0], actions[2]], action_accept:  [actions[1]]


            # States  pvc_charge_power[ids], for house E001
            if ids == "E001":
                pvc_e001_ = np.array([pvc_charge_power["E001"]])
                load_e001_ = np.array([ups_output_power["E001"]])
                p2_e001_ = np.array([p2["E001"]])
                rsoc_e001_ = np.array([rsoc["E001"]])

                all_e001_ = np.concatenate([pvc_e001_, load_e001_, p2_e001_, rsoc_e001_], axis=-1)

            if ids == "E002":
                pvc_e002_ = np.array([pvc_charge_power["E002"]])
                load_e002_ = np.array([ups_output_power["E002"]])
                p2_e002_ = np.array([p2["E002"]])
                rsoc_e002_ = np.array([rsoc["E002"]])

                all_e002_ = np.concatenate([pvc_e002_, load_e002_, p2_e002_, rsoc_e002_], axis=-1)

            if ids == "E003":
                pvc_e003_ = np.array([pvc_charge_power["E003"]])
                load_e003_ = np.array([ups_output_power["E003"]])
                p2_e003_ = np.array([p2["E003"]])
                rsoc_e003_ = np.array([rsoc["E003"]])

                all_e003_ = np.concatenate([pvc_e003_, load_e003_, p2_e003_, rsoc_e003_], axis=-1)

            if ids == "E004":
                pvc_e004_ = np.array([pvc_charge_power["E004"]])
                load_e004_ = np.array([ups_output_power["E002"]])
                p2_e004_ = np.array([p2["E004"]])
                rsoc_e004_ = np.array([rsoc["E004"]])

                all_e004_ = np.concatenate([pvc_e004_, load_e004_, p2_e004_, rsoc_e004_], axis=-1)

        # print(rsoc)
        # {'E001': 29.98, 'E002': 29.99, 'E003': 29.98, 'E004': 29.99}
        rsoc_ave_ = np.mean(rsoc_list)  # get average rsoc of this community
        # print(rsoc_ave)

        if house_id == "E001":

            state_ = np.concatenate([all_e001_, np.array([rsoc_ave_])], axis=-1)
        elif house_id == "E002":
            state_ = np.concatenate([all_e002_, np.array([rsoc_ave_])], axis=-1)
        elif house_id == "E003":
            state_ = np.concatenate([all_e003_, np.array([rsoc_ave_])], axis=-1)
        elif house_id == "E004":
            state_ = np.concatenate([all_e004_, np.array([rsoc_ave_])], axis=-1)
        else:
            print("wrong house id, input again")

        reward = p2_e001_
        # done = time.sleep(5)  # time, e.g., one hour(time.sleep(60*60)) or given #EPI

        return np.array(state_, dtype=np.float32), reward,  {}  # done

    def reset(self):
        # reset the states according to standard.json file (../apis-emulator/jsontmp)
        # all values are the same to each house
        # super().reset(seed=seed)

        # init state
        pvc_charge_power = np.array([0.])
        ups_output_power = np.array([0.])
        p2 = np.array([0.])
        rsoc = np.array([50.])
        # wg = np.array([0])
        # wb = np.array([-4.5])
        rsoc_ave = np.array([50.])  # average rsoc in the same community

        # self.state = np.array([self.state])
        self.state = np.concatenate([pvc_charge_power, ups_output_power, p2, rsoc, rsoc_ave], axis=-1)

        # return np.array(self.state, dtype=np.float32)
        return np.array(self.state, dtype=np.float32)
