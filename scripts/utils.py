import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt 

from unity import reset_brain, get_state, step

def compute_q_value(model, state):

    input_tensor = torch.from_numpy(state).double()
    model_output = model(input_tensor)
    action = np.argmax(model_output)
    
    q_val = model_output[action]

    return q_val


def dqn(env, agent, brain_name, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    env_info = reset_brain(env, brain_name, train = False)
    for i_episode in range(1, n_episodes+1):
        state = get_state(env_info)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = step(env, brain_name, action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        env_info = reset_brain(env, brain_name, train = True) 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            # break
    return scores


def plot(scores, save_path):

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    def plot(scores):

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    

    plt.title('Training Performance')
    
    # Save the plot if a save path is provided

    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save
    print(f"Plot saved to {save_path}")
