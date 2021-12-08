"""
#  for each nodes
#  get the log data from apis-emulator for states

@author: Qiong
"""

import logging.config
import time

# from main import batteryLevel

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
        # request and accept level: between [0, 1]
        self.action_request_space = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.action_accept_space = [0.5, 0.4, 0.3, 0.2, 0.1]
        # action_request ={[0.9, 0.8, 0.7, 0.6, 0.5],
        # [0.9, 0.8, 0.7, 0.6, 0.5],
        # [0.9, 0.8, 0.7, 0.6, 0.5],
        # [0.9, 0.8, 0.7, 0.6, 0.5],
        # [0.9, 0.8, 0.7, 0.6, 0.5]} list of actions, pick one of the list
        self.n_actions = len(self.action_request_space) + len(self.action_accept_space)
        self.batteryLevel_req = [" ", " ", " ", " "]
        self.batteryLevel_acc = [" ", " ", " ", " "]

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
    def step(self, action_request, action_accept):
        # batteryLevel_req = ["excess", "sufficient", "scarce", "short"]
        # batteryLevel_acc = ["excess", "sufficient", "scarce", "short"]
        batteryLevel_req = [" ", " ", " ", " "]
        batteryLevel_acc = [" ", " ", " ", " "]

        if self.action_request_space[action_request] >= 0.8:
            batteryLevel_req[0] = "excess"  # discharge
        elif 0.8 > self.action_request_space[action_request] >= 0.6:
            batteryLevel_req[1] = "sufficient"  # discharge
        elif 0.6 > self.action_request_space[action_request] >= 0.5:
            batteryLevel_req[2] = "scare"  # charge
        elif self.action_request_space[action_request] < 0.5:
            batteryLevel_req[3] = "short"  # charge

        if self.action_accept_space[action_accept] >= 0.5:
            batteryLevel_acc[0] = "excess"  # discharge
        elif 0.4 > self.action_accept_space[action_accept] >= 0.3:
            batteryLevel_acc[1] = "sufficient"  # discharge
        elif 0.3 > self.action_accept_space[action_accept] >= 0.2:
            batteryLevel_acc[2] = "scare"  # charge
        elif self.action_accept_space[action_accept] < 0.2:
            batteryLevel_acc[3] = "short"  # charge

        # minimize purchase from the powerline
        # receiving states: pv , load, p2, rsoc
        # powerline_energy = power_flow_to_battery - load ?
        # reward = powerline_energy
        # reward = p2

        #  return next_s, reward
        print(batteryLevel_req, batteryLevel_acc)
        return batteryLevel_req, batteryLevel_acc  # , reward

    # def reset(self):

    # def step(self): # request, accept

    # reward function
    # reward = -cost

    def CreateSce(self, action_request, action_accept, batteryLevel_req, batteryLevel_acc):
        # batteryLeve, init actions
        # batteryLevel = ["excess", "sufficient", "scarce", "short"]
        # newSce = CreateScenario(batteryLevel=self.batteryLevel, action=action)
        newSce = CreateScenario(action_request=action_request, action_accept=action_accept,
                                batteryLevel_req=batteryLevel_req, batteryLevel_acc=batteryLevel_acc)
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
