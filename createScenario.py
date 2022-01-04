# create by Qiong
# create scenario files for updating the energy exchange rules

import json
import time
import os
import numpy as np


def print_ts(message):
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), time.localtime()), message)


def run(interval, command):
    print_ts("-" * 100)
    # print_ts("Command %s" % command)
    print_ts("starting every %s seconds." % interval)
    print_ts("-" * 100)

    while True:
        try:
            # sleep for the remaining seconds of interval
            time_remain = interval - time.time() % interval
            print_ts("sleep until %s (%s seconds..." % ((time.ctime(time.time() + time_remain)), time_remain))
            time.sleep(time_remain)
            print_ts("starting command.")
            # execute the command
            status = os.system(command)
            print_ts("-" * 100)
            print_ts("command status = %s." % status)
        except Exception as e:
            print(e)


# generate scenario.json file
# t_end = time.time() + 60 * 15  # run for 15 min x 60 s = 900 seconds.
# while time.time() < t_end:
#     print(time.time())
#
#     with open('scenario.json', 'w') as jsonfile:
#         json.dump(data1, jsonfile)
#
#     print("scenario file updated")
#     # refresh every 5 seconds
#     time.sleep(5)
# with open('scenario.json', 'w') as jsonfile:
#     json.dump(data1, jsonfile)


# create scenario file and put it under the dir of apis-main/exe

class CreateScenario():
    def __init__(self,  action_request, action_accept):

        # self.action_space = [0.8, 0.5, 0.4]
        self.action_request_space = np.linspace(0.2, 0.9, 8).tolist()
        self.action_accept_space = np.linspace(0.2, 0.9, 8).tolist()
        # set time periods for scenario files
        # self.timePeriods = ["00:00:00-12:00:00", "12:00:00-24:00:00"]
        self.timePeriods = ["00:00:00-24:00:00"]
        # per hour: TimePeriods[0],...,TimePeriods[23]
        self.TimePeriods = ["00:00:00-01:00:00", "01:00:00-02:00:00", "02:00:00-03:00:00",
                            "03:00:00-04:00:00", "04:00:00-05:00:00", "05:00:00-06:00:00",
                            "06:00:00-07:00:00", "07:00:00-08:00:00", "08:00:00-09:00:00",
                            "09:00:00-10:00:00", "10:00:00-11:00:00", "11:00:00-12:00:00",
                            "12:00:00-13:00:00", "13:00:00-14:00:00", "14:00:00-15:00:00",
                            "15:00:00-16:00:00", "16:00:00-17:00:00", "17:00:00-18:00:00",
                            "18:00:00-19:00:00", "19:00:00-20:00:00", "20:00:00-21:00:00",
                            "21:00:00-22:00:00", "22:00:00-23:00:00", "23:00:00-24:00:00"]
        self.batterySize = 4800
        # batteryLevel : 4 levels
        self.batteryLevel = ["excess", "sufficient", "scarce", "short"]
        self.data = {
            "#": "place this file at the path defined by 'scenarioFile' in config file",
            "refreshingPeriodMsec": 5000,

            "acceptSelection": {
                "strategy": "pointAndAmount"
            },

            self.timePeriods[0]: {
                "batteryStatus": {  # batteryLevels
                    # list of actions
                    str(self.batterySize * self.action_request_space[action_request[0]]) + "-": self.batteryLevel[0],
                    str(str(self.batterySize * self.action_accept_space[action_accept[0]]) + "-" + str(self.batterySize * self.action_request_space[action_request[0]])): self.batteryLevel[1],
                    str(str(self.batterySize * self.action_request_space[action_request[1]]) + "-" + str(self.batterySize * self.action_accept_space[action_accept[0]])): self.batteryLevel[2],
                    "-" + str(self.batterySize * self.action_request_space[action_request[1]]): self.batteryLevel[3]
                },
                "request": {
                    self.batteryLevel[0]: {"discharge": {
                        "limitWh": self.batterySize * self.action_request_space[action_request[0]],  # 0.8,
                        "pointPerWh": 10
                    }},
                    self.batteryLevel[1]: {},
                    self.batteryLevel[2]: {},
                    self.batteryLevel[3]: {"charge": {
                        "limitWh": self.batterySize * self.action_request_space[action_request[1]],  # 0.4,
                        "pointPerWh": 10
                    }}
                },
                "accept": {
                    self.batteryLevel[0]: {"discharge": {
                        "limitWh": self.batterySize * self.action_accept_space[action_accept[0]],  # 0.5,
                        "pointPerWh": 10
                    }},
                    self.batteryLevel[1]: {"discharge": {
                        "limitWh": self.batterySize * self.action_accept_space[action_accept[0]],  # 0.5,
                        "pointPerWh": 10
                    }},
                    self.batteryLevel[2]: {"charge": {
                        "limitWh": self.batterySize * self.action_accept_space[action_accept[0]],  # 0.5,
                        "pointPerWh": 10
                    }},
                    self.batteryLevel[3]: {"charge": {
                        "limitWh": self.batterySize * self.action_accept_space[action_accept[0]],  # 0.5,
                        "pointPerWh": 10
                    }}
                }
            },

            # self.timePeriods[1]: {
            #     "batteryStatus": {
            #         str(self.batterySize * 0.7) + "-": self.batteryLevel[0],
            #         str(str(self.batterySize * 0.5) + "-" + str(self.batterySize * 0.7)): self.batteryLevel[1],
            #         str(str(self.batterySize * 0.3) + "-" + str(self.batterySize * 0.5)): self.batteryLevel[2],
            #         "-" + str(self.batterySize * 0.3): self.batteryLevel[3]
            #     },
            #     "request": {
            #         self.batteryLevel[0]: {"discharge": {
            #             "limitWh": self.batterySize * 0.7,
            #             "pointPerWh": 10
            #         }},
            #         self.batteryLevel[1]: {},
            #         self.batteryLevel[2]: {},
            #         self.batteryLevel[3]: {"charge": {
            #             "limitWh": self.batterySize * 0.3,
            #             "pointPerWh": 10
            #         }}
            #     },
            #     "accept": {
            #         self.batteryLevel[0]: {"discharge": {
            #             "limitWh": self.batterySize * 0.5,
            #             "pointPerWh": 10
            #         }},
            #         self.batteryLevel[1]: {"discharge": {
            #             "limitWh": self.batterySize * 0.5,
            #             "pointPerWh": 10
            #         }},
            #         self.batteryLevel[2]: {"charge": {
            #             "limitWh": self.batterySize * 0.5,
            #             "pointPerWh": 10
            #         }},
            #         self.batteryLevel[3]: {"charge": {
            #             "limitWh": self.batterySize * 0.5,
            #             "pointPerWh": 10
            #         }}
            #     }
            # }

        }

        # TODO: different files for different agent
        # try : different class for different agents
        self.filename1 = "scenario.json"
        self.filename2 = "scenario2.json"
        self.filename3 = "scenario3.json"
        self.filename4 = "scenario4.json"
        # self.desired_dir = "/home/doya/Documents/APIS/apis-main/exe/"
        # self.desired_dir = "/Users/Huang/Documents/APIS/apis-main/exe/"
        # get the parent path -> ../APIS
        self.getpath = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        self.desired_dir = self.getpath + "/apis-main/exe/"
        self.full_path1 = os.path.join(self.desired_dir, self.filename1)
        self.full_path2 = os.path.join(self.desired_dir, self.filename2)
        self.full_path3 = os.path.join(self.desired_dir, self.filename3)
        self.full_path4 = os.path.join(self.desired_dir, self.filename4)

    def write_json(self):
        with open(self.full_path1, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        with open(self.full_path2, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        with open(self.full_path3, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        with open(self.full_path4, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


# if __name__ == "__main__":
#     interval = 60 * 60  # every 60 * 60s
#     command = createJson()
#     run(interval, command)
