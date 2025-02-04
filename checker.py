
import os
import importlib
# import torch
import gym
import sys
import numpy as np
import inspect
from osim.env import L2M2019Env

def check_file_integrity(foldernames):

    checking_list = []
    for file in foldernames:
        if not os.path.isdir(file) and "_hw4" in file:
            checking_list.append(file.split("_")[2])
 
    elements = ["data","test.py","train.py"]
    
    for name in elements:
        if name not in checking_list:
            print(f"\033[91m hw4_{name} IS MISSING\033[0m")
            return False
    
    return True
        


def check_agent(foldernames):

    test_file = None
    for file in foldernames:
        if "hw4_test.py" in file:
            test_file = file

    module_name = test_file.replace('/', '.').replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        Agent = getattr(module, 'Agent')
        return Agent
    except AttributeError as e:
        return None
    

        
def check_agent_act(Agent):

    if hasattr(Agent, 'act') and callable(getattr(Agent, 'act')):

        # cheking data form
        act_signature = inspect.signature(Agent.act)
        params = list(act_signature.parameters.keys())

        act_signature = inspect.signature(Agent.__init__)
        params_i = list(act_signature.parameters.keys())

        if len(params_i) == 1 and params[0] == 'self':
            print(f"\033[92mAgent Init Function Checking: PASS \033[0m")
        else:
            print(f"\033[91mAgent Init FUNCTION HAS WRONG PARAMETER LIST!\033[0m")
            return False

        if len(params) == 2 and params[1] == 'observation':
            print(f"\033[92mAgent Act Function Checking: PASS +5\033[0m")
        else:
            print(f"\033[91mAgent ACT FUNCTION HAS WRONG PARAMETER LIST!\033[0m")
            return False
    else:
        print(f"\033[91mAgent MISSING ACT FUNCTION!\033[0m")
        return False

    

    return True


def check_data_form(Agent):

    env = L2M2019Env(visualize=False,difficulty=2)
    test_obs = env.reset()

    agent = Agent() 

    action_space = gym.spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)

    action = agent.act(test_obs)
    
    # if not action_space.contains(action):
        
    #     print(f"\033[91mAgent ACT FUNCTION RETURNS ACTION OUT OF BOUNDS\033[0m")        
    #     # print(f'action: {np.all(action < 1) and np.all(action > 0)}')
    #     print(f'action: {action.shape}')
    #     return False
        


    # try:
    #     action = agent.act(test_obs)

    #     if not action_space.contains(action):
    #         print(f"\033[91mAgent ACT FUNCTION RETURNS ACTION OUT OF BOUNDS\033[0m")
    #         return False
    # except Exception as e:
    #     print(e)
    #     print(f"\033[91mAgent ACT FUNCTION CANNOT HANDLE CORRECT OBSERVATION FORM\033[0m")
    #     return False

    print(f"\033[92mData Form Checking: PASS +5\033[0m")
    return True

# main function
def main():

    score = 0

    files = []
    foldernames = os.listdir()
    for file in foldernames:
        if file.startswith("1"):
            files.append(file)

    # checking
    if check_file_integrity(files):
        print(f"\033[92mFile Checking: PASS +5\033[0m")
        score += 5
    else:
        return

    
    Agent = check_agent(files)
    if Agent == None:
        print(f"\033[91mAgent Class IS MISSING IN TEST FILE\033[0m")
    else:
        print(f"\033[92mAgent Class Checking: PASS +5\033[0m")
        score += 5

    if check_agent_act(Agent):
        score += 5

    if check_data_form(Agent):
        score += 5

    # showing score
        
    print(f"\033[93mScore: {score}/20\033[0m")
    
 
          

if __name__ == "__main__":
    main()
