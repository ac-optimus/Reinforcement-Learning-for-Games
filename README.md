# Intro
In this repository, we hope to try and analyze some popular ways that are used to train agents in the domain of Reinforcement Learning. Initially, we tried having our environment and using it, but since we wanted to move to a more complex environment, we ended up using [OpenAI's Gym][1]. <br>
After working on our environment, we tried using Q-Learning on [Berkley AI Course][2]'s Pacman Environment.


## Files
- `personal_game.ipynb`: our take on creating an environment to understand rewards, states, actions, etc. better.<br>
- `DQNlearning.py`: file that tries to train an agent to play `CartPool-v1` from OpenAI gym using the observation space vector and a fully connected neural network.<br> The code uses the base code at [gsurma/cartpole][3] and is ported to use PyTorch with some experimentations on the network architecture.
- `classicQ.py`: trying to run `CartPool-v1` with the help of classic Q Learning algorithm by discretizing the continuous observation space. Inspired from [`qcartpole.py`][6].

We plan to look at the following policies to try to train our agents:
- [x] Q-learning on Berkley Pacman
- [x] Q-learning on Cartpool
- [x] Deep Q Network directly using observation space vector (if available)
- [ ] Deep Q Network using convolutions on the render of the environment
- [ ] Double Q Learning


## Plan

Plan to use Deep Q Network using CNNs on CartPool-v1 and add some graphs showcasing the difference in the speed of learning.


## Citations

- [OpenAI's Gym][1]
- [PyTorch][4]
- [Berkley AI Course][2]
- [gsurma/cartpole][3]
- [Pytorch Dual DQN][5]
- [Classical Q Learning on Cartpool][6]

  [1]: https://gym.openai.com/
  [2]: http://ai.berkeley.edu/home.html
  [3]: https://github.com/gsurma/cartpole
  [4]: https://pytorch.org/
  [5]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
  [6]: https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578#file-qcartpole-py-L24
