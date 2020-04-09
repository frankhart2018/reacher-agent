# Reacher Agent Testing for a single agent

# Import libraries
import torch
from unityagents import UnityEnvironment
import numpy as np

# Import the Actor model
from model import Actor

# Create instance of Reacher environment
env = UnityEnvironment(file_name='Reacher.app') # Update the app name/location if not using macOS

# Get brain

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Load Actor model weights
actor = Actor(state_size=33, action_size=4, seed=0)
actor.load_state_dict(torch.load('checkpoint_actor.pth'))

# Testing
def test(state):

    """
    Testing the Reacher agent for a single agent

    Params
    ======
        state (numpy.ndarray): Current state that the agent is experiencing
    """

    global actor

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        qnetwork = qnetwork.cuda()

    state = torch.from_numpy(state).float().to(device)
    actor.eval()
    with torch.no_grad():
        action = actor(state).cpu().data.numpy()

    return np.clip(action, -1, 1)

env_info = env.reset(train_mode=False)[brain_name]       # reset the environment
state = env_info.vector_observations[0]
score = 0.0
for t in range(1000):
    action = test(state)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    state = next_state
    score += reward
    if done:
        break
print("Score:", score)
env.close()
