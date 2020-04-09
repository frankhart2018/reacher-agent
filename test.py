# Reacher Agent Testing for twenty (20) agents

# Import libraries
import torch
from unityagents import UnityEnvironment
import numpy as np

# Import the Actor model
from model import Actor

# Create instance of Reacher environment
env = UnityEnvironment(file_name='Reacher-20.app') # Update the app name/location if not using macOS

# Get brain

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Load Actor model weights
actor = Actor(state_size=33, action_size=4, seed=0)
actor.load_state_dict(torch.load('checkpoint_actor.pth'))

# Testing
def test(state, agents, action_size):

    """
    Testing the Reacher agent for a single agent

    Params
    ======
        state (numpy.ndarray): Current state that the agents are experiencing
        agents (int):          The number of agents (= 20 in this case)
        action_size (int):     Number of possible actions an agent can take
    """

    global actor

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        qnetwork = qnetwork.cuda()

    state = torch.from_numpy(state).float().to(device)
    action = np.zeros((agents, action_size))
    actor.eval()
    with torch.no_grad():
            for agent, s in enumerate(state):
                action[agent,:] = actor(s).cpu().data.numpy()

    return np.clip(action, -1, 1)

env_info = env.reset(train_mode=False)[brain_name]       # reset the environment
state = env_info.vector_observations
score = np.zeros(20)
for t in range(1000):
    action = test(state, agents=20, action_size=4)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations
    reward = env_info.rewards
    done = env_info.local_done
    state = next_state
    score += reward
    if np.any(done):
        break
print("Score:", np.mean(score))
env.close()
