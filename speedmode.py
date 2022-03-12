
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import logging.config

logger = logging.getLogger(__name__)
import time
import global_var as gl
import analyser
import core
import config as conf
import requests, json
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


# def indivToMemory():
for key, val in gl.oesunits.items():
    l = []
    l.append(key)
    l.append(str(gl.now))
    l.append(val["oesunit"]["display"])
    l.append(val["emu"]["rsoc"])
    l.append(val["emu"]["pvc_charge_power"])
    l.append(val["emu"]["ups_output_power"])
    l.append(val["emu"]["charge_discharge_power"])
    l.append(val["dcdc"]["meter"]["wg"])
    l.append(val["dcdc"]["powermeter"]["p2"])
    l.append(val["dcdc"]["meter"]["ig"])

# print(l)


