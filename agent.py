"""
#  for each nodes
#  get the log data from apis-emulator for states

@author: Qiong
"""

import logging.config
import time

logger = logging.getLogger(__name__)

import global_var as gl
import config as conf

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from createScenario import CreateScenario

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
        self.action_space = [0.8, 0.5, 0.4]
        self.n_actions = len(self.action_space)

    """
    def _build_agent(self, action, rsoc):
        if rsoc >= 80.:
            action == "excess"
        elif 50. <= rsoc < 80.:
            action == "sufficient"
        elif 40. <= rsoc < 50.:
            action == "scare"
        else:  # rsoc < 40.
            action == "short"
    """

    # actions 0.8, 0.5, 0.4 \in [0,1], list of possible actions
    # reward
    def step(self, action):
        if self.action_space[action] == 0.8:
            self.batteryLevel == "excess"
        elif self.action_space[action] == 0.5:
            self.batteryLevel == "sufficient"
        elif self.action_space[action] == 0.4:
            self.batteryLevel == "scare"
        else:
            self.batteryLevel == "short"


        # minimize purchase from the powerline
        # receiving states: pv , load, p2, rsoc
        # powerline_energy = power_flow_to_battery - load ?
        # reward = powerline_energy
        # reward = p2

        #  return next_s, reward
        return self.batteryLevel  # , reward

    # def reset(self):

    # def step(self): # request, accept

    # reward function
    # reward = -cost

    def CreateSce(self, action):
        # batteryLeve, init actions
        # batteryLevel = ["excess", "sufficient", "scarce", "short"]
        # newSce = CreateScenario(batteryLevel=self.batteryLevel, action=action)
        newSce = CreateScenario(action=action)
        # newSce.batteryLevel
        newSce.write_json()

        # if __name__ == "__main__":
        #     interval = 60 * 60  # every 60 * 60s
        #     command = createJson()
        #     run(interval, command)

# action section
# rsoc = []
# action = {}
#
# if rsoc >= 80.:
#     action[item] == "excess"
# elif 50. <= rsoc < 80.:
#     action[item] == "sufficient"
# elif 40. <= rsoc < 50.
#     action[item] == "scare"
# else: # rsoc < 40.
#     action[item] == "short"
