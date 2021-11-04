#  for each nodes
#  get the log data from apis-emulator for states
#  created by Qiong

"""
Create on Sep 26, 2021

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
#  create new scenario file and put it under apis-main dir
# timePeriods = ["00:00:00-12:00:00", "12:00:00-24:00:00"]
# batterySize = 4800
# data1 = {
#     "#": "place this file at the path defined by 'scenarioFile' in config file",
#     "refreshingPeriodMsec": 5000,
#
#     "acceptSelection": {
#         "strategy": "pointAndAmount"
#     },
#
#     timePeriods[0]: {
#         "batteryStatus": {
#             str(batterySize * 0.8) + "-": "excess",
#             str(str(batterySize * 0.5) + "-" + str(batterySize * 0.8)): "sufficient",
#             str(str(batterySize * 0.4) + "-" + str(batterySize * 0.5)): "scarce",
#             "-" + str(batterySize * 0.4): "short"
#         },
#         "request": {
#             "excess": {"discharge": {
#                 "limitWh": batterySize * 0.8,
#                 "pointPerWh": 10
#             }},
#             "sufficient": {},
#             "scarce": {},
#             "short": {"charge": {
#                 "limitWh": batterySize * 0.4,
#                 "pointPerWh": 10
#             }}
#         },
#         "accept": {
#             "excess": {"discharge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }},
#             "sufficient": {"discharge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }},
#             "scarce": {"charge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }},
#             "short": {"charge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }}
#         }
#     },
#
#     timePeriods[1]: {
#         "batteryStatus": {
#             str(batterySize * 0.7) + "-": "excess",
#             str(str(batterySize * 0.5) + "-" + str(batterySize * 0.7)): "sufficient",
#             str(str(batterySize * 0.3) + "-" + str(batterySize * 0.5)): "scarce",
#             "-" + str(batterySize * 0.3): "short"
#         },
#         "request": {
#             "excess": {"discharge": {
#                 "limitWh": batterySize * 0.7,
#                 "pointPerWh": 10
#             }},
#             "sufficient": {},
#             "scarce": {},
#             "short": {"charge": {
#                 "limitWh": batterySize * 0.3,
#                 "pointPerWh": 10
#             }}
#         },
#         "accept": {
#             "excess": {"discharge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }},
#             "sufficient": {"discharge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }},
#             "scarce": {"charge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }},
#             "short": {"charge": {
#                 "limitWh": batterySize * 0.5,
#                 "pointPerWh": 10
#             }}
#         }
#     }
# }
newSce = CreateScenario()
newSce.write_json()

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