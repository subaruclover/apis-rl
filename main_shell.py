#!/usr/bin/env python
"""
run the main.py file for multiple times (runs)

Author: Qiong
"""

import os

N_RUN = 3
run = 0
while run < 3:
    # execute the code for N_RUN times
    os.system("python main3.py")
    run += 1
