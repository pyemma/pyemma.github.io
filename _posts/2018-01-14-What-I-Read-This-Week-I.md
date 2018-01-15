---
  layout: single
  title: What I Read This Week 1
---
### [The 3 Tricks That Made AlphaGo Zero Work](https://hackernoon.com/the-3-tricks-that-made-alphago-zero-work-f3d47b6686ef)
This post explains why AlphaGo Zero out-perform than it's elder brother AlphaGo, summarizing in 3 points that lead to the supreme result:
* Use the evaluations provided by MCTS to continually improve the neural network's evaluations of the board position, instead of using human play (This is actually the idea of using **better training sample**).
* Use a single neural network to predict which **move** to recommend *and* which **move** are likely to win the game (This is the idea of using [**Multitask Learning**](http://ruder.io/multi-task/index.html#introduction)).
* Use a upgrade version of neural network (from convolutional neural network to **residual neural network**).

### [Intuitive RL: Intro to Advantage-Actor-Critic (A2C)](https://medium.com/@rudygilman/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752)
A very vivid introduction of **Advantage-Actor-Critic** reinforcement learning.
