# Reacher Agent Training

# Import libraries
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt

# Create instance of Reacher environment
env = UnityEnvironment(file_name='Reacher-20.app') # Update the app name/location if not using macOS

# Get brain, number of agents, size of state, size of action

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# Import the Deep Deterministic Policy Gradients (DDPG) Agent
from ddpg_agent import Agent

agent = Agent(state_size=state_size, action_size=action_size, agents=num_agents, random_seed=0)

# Training loop
def ddpg(n_episodes=2000, max_t=1000, print_every=10):

    """
    Training loop for the DDPG (Deep Deterministic Policy Gradients) Agent

    Params
    ======
        n_episodes (int):  The number of episodes to run for the agent to learn
        max_t (int):       Maximum number of time steps which the agent can experience in a single episode
        print_every (int): Print average score every 'n' episodes
    """

    scores_deque = deque(maxlen=print_every)
    max_score = -np.Inf
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        state = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque),
                                                                          np.mean(score)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100,
                                                                                         np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores

scores = ddpg()

# Plot the average score during training
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
