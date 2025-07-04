# Soft-Actor-Critic-PyTorch
An implementation of Soft Actor Critic using PyTorch. This implementation doesn't include automatic temperature optimization (which could be a future work). It was trained for a 1000 games (or 1 million steps) on OpenAI Gymnasium Mujoco environment Half-Cheetah-v5.

## Running Average Plot for Half-Cheetah
It can be observed that the average reward return curve had not yet saturated and the agent can be improved further upon training for a few thousand more games.

![](https://github.com/DishankJ/Soft-Actor-Critic-PyTorch/blob/main/saved/sac_half_cheetah.png?raw=true)

## Simulation GIF
![](https://github.com/DishankJ/Soft-Actor-Critic-PyTorch/blob/main/saved/cheetah.gif?raw=true)
