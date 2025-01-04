import numpy as np

from unity import *
from config import FILE_PATH, EPOCHS, EPISODE_LEN, ALPHA, BATCH_SIZE

from modelling import Agent
from utils import dqn, plot

env = get_environment(FILE_PATH)

brain_name = get_brain_name(env)
brain = get_brain(env, brain_name)
env_info = reset_brain(env, brain_name, train = False)

INPUT = len(get_state(env_info))
OUTPUT = get_actions(brain)

cur_state = get_state(env_info)

agent = Agent(INPUT, OUTPUT)

scores = dqn(env, agent, brain_name)
plot(scores, save_path = "reports/report.png")

close_env(env)