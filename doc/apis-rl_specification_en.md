**apis-rl Specification Document**

# **Contents**
- [**Contents**](#contents)
- [**1. Terms and abbreviations**](#1-terms-and-abbreviations)
- [**2. Overview**](#2-overview)
- [**3. Functions**](#3-functions)
  - [**3.1. config**](#31-config)
  - [**3.2. global_var**](#32-global_var)
  - [**3.3. agent**](#33-agent)
    - [**3.3.1. createScenario**](#331-createScenario)
    - [**3.3.2. action**](#332-action)
  - [**3.4. RL_learn**](#34-rl_learn)
  - [**3.5. main**](#35-main)
<br>

# **1. Terms and abbreviations**

| **Term**  | **Explanation**                                                                                 |
| --------- | ----------------------------------------------------------------------------------------------- |
| apis-rl   | Software for energy sharing with reinforcement learning methods                                 |
| EMU       | Energy Management Unit: A device for controlling a power storage system.                        |

<br>

# **2. Overview**

The Emulator runs a computer emulation that reproduces the hardware system for energy sharing, including the battery system and the DC/DC converter, etc. The Emulator reads in data on the amount of solar radiation and the power consumption of residences and emulates the flow of energy such as the power generated and consumed by multiple residences, and battery system charging and discharging. The emulation conditions can be changed in real time by using a Web browser to access and change the hardware parameters. There is also a function for communication with apis-main, which reads storage battery data from the hardware emulation on the computer and operates the DC/DC converter to emulate energy sharing.


# **3. Functions**
    
## **3.1. config**

## **3.2. global_var**
  
## **3.3. agent**

![](media/media/states.gif)
Figure 3-1

![](media/media/refresh%20scenario.png)
Figure 3-2

### **3.3.1. createScenario**
Create scenario files for updating the energy exchange rules. The class CreateScenario()
create scenario file and put it under the dir of apis-main/exe
<br>

### **3.3.2. action**
For scenarios, the key decision value for each node to set the thresholds of
different battery status are the boundary value of different situations, i.e., 
"excess", "sufficient", "scarce", "short"。

The design of initial values for the actions are set to be 0.8, 0.5, and 0.4.
<p>Actions: &isin; [0,1] </p>


## **3.4. RL_learn**

## **3.5. main**
Each node has its own main.py function file to run its DQN methods.
main.py main2.py main3.py main4.py etc....



