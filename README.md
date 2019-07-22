# Intro
In this project, we analyze some popular ways that are used to train agents in the domain of Reinforcement Learning. Initially, we tried having our own environment and using it, but since we wanted to move to a more complex environment, we ended up using [OpenAI's Gym][1]. <br>
After working on our environment, we tried using Q-Learning on [Berkley AI Course][2]'s Pacman Environment.

## We looked at the following policies to try to train our agents:
- [x] Q-learning on Berkley Pacman
- [x] Q-learning on Cartpool
- [x] Deep Q Network directly using observation space vector
- [x] Deep Q Network using convolutions on the render of the environment

- [x] Double Q Learning


## Final Results

A detailed report of our work can be found in [final_report.prf](final_report.pdf)


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
