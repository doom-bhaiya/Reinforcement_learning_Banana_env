import numpy as np

from unity import *
from config import FILE_PATH

env = get_environment(FILE_PATH)

brain_name = get_brain_name(env)
reset_brain(env, brain_name, train = False)
for i in range(100):
    step(env, brain_name, 3)

input("Enter something to close :")

close_env(env)