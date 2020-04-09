# Report of training the agent

<p>The model was trained for a maximum of 2,000 episodes but the agent was able to solve the environment (i.e. get atleast +30 average score over 100 adjacent episodes).</p>

## Model

<p align="justify">The Actor Network has three dense (or fully connected layers). The first two layers have <b>400 and 300</b> nodes respectively activated with <b>ReLU</b> activation function. The final (output layer) has <b>4</b> nodes and is activated with tanh activation. This network takes in as input the <b>33</b> dimensional current state and gives as output <b>4</b> to provide the action at current state that the agent is supposed to take.</p>

<p align="justify">The Critic Network has three dense (or fully connected layers). The first two layers have <b>404 and 300</b> nodes respectively activated with <b>ReLU</b> activation function. The final (output layer) has <b>4</b> nodes and is activated with linear activation (no activation at all). This network takes in as input the <b>33</b> dimensional current state and <b>4</b> dimensional action and gives as output a single real number to provide the Q-value at current state and action taken in that state.</p>

<p>Both the neural networks used Adam optimizer and Mean Squared Error (MSE) as the loss function.</p>

<p>The following image provides a pictorial representation of the Actor Network model:</p>

<p align='center'>
  <img src='images/actor-network.png' alt='Pictorial representation of Q-Network'>
</p>

<p>The following image provides a pictorial representation of the Critic Network model:</p>

<p align='center'>
  <img src='images/critic-network.png' alt='Pictorial representation of Q-Network'>
</p>

<p>The following image provides the plot for score v/s episode number:</p>

<p align='center'>
  <img src='images/plot.png' alt='Plot for score v/s episode number' width='650'>
</p>

## Performance

<p>The model was trained on MacBook Air 2017 with 8GB RAM and Intel Core i5 Processor.</p>

<ul>
  <li><b>Number of episodes required to solve the environment</b>: -37 episodes</li>
  <li><b>Final score of the agent</b>: 30.57</li>
</ul>

## Hyperparameters used

| Hyperparameter           | Value  | Description                                               |
|--------------------------|--------|-----------------------------------------------------------|
| Buffer size              | 100000 | Maximum size of the replay buffer                         |
| Batch size               | 128    | Batch size for sampling from replay buffer                |
| Gamma (<b>γ</b>)         | 0.99   | Discount factor for calculating return                    |
| Tau (<b>τ</b>)           | 0.001  | Hyperparameter for soft update of target parameters       |
| Learning Rate Actor      | 0.0003 | Learning rate for the actor neural network                |
| Learning Rate Critic     | 0.001  | Learning rate for the critic neural network               |

## Future work

<p>The following algorithms can be considered for further development of this agent:</p>

<ul>
  <li>Proximal Policy Optimization (PPO)</li>
  <li>Generalized Advantage Estimation (GAE)</li>
  <li>Advantage Actor-Critic (A2C)</li>
  <li>Asynchronous Advantage Actor-Critic (A3C)</li>
</ul>