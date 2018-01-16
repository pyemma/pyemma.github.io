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
1. First we should keep in mind that **Actor-Critic** is a blend of both value estimation and policy estimation reinforcement learning method, a.k.a we will try to learn value function, as well as policy from game play (This is different from pure value function based method and policy based method).
2. In **Actor-Critic**, the **Actor** will tries to optimize the parameter for policy and **Critic** will tries to optimize the parameter for the value function of a state. This can be done by having a single model outputting both the value of the state, as will as the probability of action.
3. By jump into one state, taking action and get reward. We will get the training examples for our **Critic**. The estimate for each state will become more and more accurate. In this way, we don't need to wait until the end of the game to get the value of each state, which is high in variance.
4. In stead of simply policy gradient update the policy (which tries to avoid the action that lead to a state with low value), we use **Advantage**, which is the relative improvement of the action take (e.g. current state is -100, and take action A we arrive in a state with -20, the improvement of the action is 80!). The idea behind this is that the action might be the result that result in a low value.

### [AI and Deep Learning in 2017 – A Year in Review](http://www.wildml.com/2017/12/ai-and-deep-learning-in-2017-a-year-in-review/)
A really awesome post that summarize what is going on in deep learning in 2017. Some points that I enjoy most:
* Evolution Algorithm (e.g. Genetic Algorithm) is coming back again.
* Lots of deep learning framework is available right now: PyTorch is pretty popular in academic, but personally I thing TensorFlow is still the bests to try out (It's also my plan to be more familiar with TensorFlow and work on some side project).
* A good online reinforcement learning algorithm to read: [*OpenAI Baseline*](https://github.com/openai/baselines).
* A good online courses: https://stats385.github.io/
