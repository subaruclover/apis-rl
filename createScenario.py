# create by Qiong
# create scenario files for updating the energy exchange rules

import json
import time
import os


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


# set time periods for scenario files
timePeriods = ["00:00:00-12:00:00", "12:00:00-24:00:00"]

# per hour: TimePeriods[0],...,TimePeriods[23]
TimePeriods = ["00:00:00-01:00:00", "01:00:00-02:00:00", "02:00:00-03:00:00",
               "03:00:00-04:00:00", "04:00:00-05:00:00", "05:00:00-06:00:00",
               "06:00:00-07:00:00", "07:00:00-08:00:00", "08:00:00-09:00:00",
               "09:00:00-10:00:00", "10:00:00-11:00:00", "11:00:00-12:00:00",
               "12:00:00-13:00:00", "13:00:00-14:00:00", "14:00:00-15:00:00",
               "15:00:00-16:00:00", "16:00:00-17:00:00", "17:00:00-18:00:00",
               "18:00:00-19:00:00", "19:00:00-20:00:00", "20:00:00-21:00:00",
               "21:00:00-22:00:00", "22:00:00-23:00:00", "23:00:00-24:00:00"]
# battery Size: 4800
batterySize = 4800

data1 = {
    "refreshingPeriodMsec": 5000,

    "acceptSelection": {
        "strategy": "pointAndAmount"
    },

    timePeriods[0]: {
        "batteryStatus": {
            str(batterySize * 0.8) + "-": "excess",
            str(str(batterySize * 0.5) + "-" + str(batterySize * 0.8)): "sufficient",
            str(str(batterySize * 0.4) + "-" + str(batterySize * 0.5)): "scarce",
            "-" + str(batterySize * 0.4): "short"
        },
        "request": {
            "excess": {"discharge": {
                "limitWh": batterySize * 0.8,
                "pointPerWh": 10
            }},
            "sufficient": {},
            "scarce": {},
            "short": {"charge": {
                "limitWh": batterySize * 0.4,
                "pointPerWh": 10
            }}
        },
        "accept": {
            "excess": {"discharge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }},
            "sufficient": {"discharge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }},
            "scarce": {"charge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }},
            "short": {"charge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }}
        }
    },

    timePeriods[1]: {
        "batteryStatus": {
            str(batterySize * 0.7) + "-": "excess",
            str(str(batterySize * 0.5) + "-" + str(batterySize * 0.7)): "sufficient",
            str(str(batterySize * 0.3) + "-" + str(batterySize * 0.5)): "scarce",
            "-" + str(batterySize * 0.3): "short"
        },
        "request": {
            "excess": {"discharge": {
                "limitWh": batterySize * 0.7,
                "pointPerWh": 10
            }},
            "sufficient": {},
            "scarce": {},
            "short": {"charge": {
                "limitWh": batterySize * 0.3,
                "pointPerWh": 10
            }}
        },
        "accept": {
            "excess": {"discharge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }},
            "sufficient": {"discharge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }},
            "scarce": {"charge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }},
            "short": {"charge": {
                "limitWh": batterySize * 0.5,
                "pointPerWh": 10
            }}
        }
    }
}


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
    def __init__(self):
        self.filename = "scenario.json"
        self.desired_dir = "/home/doya/Documents/APIS/apis-main/exe/"
        self.full_path = os.path.join(self.desired_dir, self.filename)

    def write_json(self, new_data):
        with open(self.full_path, 'w') as f:
            json_string = json.dumps(new_data)
            f.write(json_string)

# write_json(new_data=data1, filename="scenario.json")

# def createJson():
#     with open('scenario.json', 'w') as jsonfile:
#         json.dump(data1, jsonfile)


# command = createJson()
#
# if __name__ == "__main__":
#     interval = 60 * 60  # every 60s
#     command = createJson()
#     run(interval, command)
