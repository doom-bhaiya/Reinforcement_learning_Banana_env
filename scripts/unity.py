from unityagents import UnityEnvironment
import numpy as np


def get_environment(file_name):

    env = UnityEnvironment(file_name)
    return env

def get_brain_name(env):

    brain_name = env.brain_names[0]
    return brain_name

def get_brain(env, brain_name):
    return env.brains[brain_name]

def reset_brain(env, brain_name, train = True):
    
    env_info = env.reset(train_mode=train)[brain_name]
    return env_info

def get_actions(brain):
    
    return brain.vector_action_space_size

def get_state(env_info):
    
    return env_info.vector_observations[0]

def get_num_agents(env_info):
    
    return env_info.agents

def step(env, brain_name, action):

    action = action.astype("int32")

    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]  
    done = env_info.local_done[0] 

    return next_state, reward, done

def close_env(env):
    env.close()