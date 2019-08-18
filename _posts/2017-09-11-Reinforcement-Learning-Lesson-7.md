---
  layout: single
  title: "Reinforcement Learning Lesson 7"
---
In the pervious notes, we are all using **model-free** reinforcement learning method to find the solution for the problem. Today we are going to introduce method that directly learns from the experience and tries to understand the underlaying world.

From [Lesson 1](http://pyemma.github.io/posts/Reinforcement-Learning-Lesson-1) we know that a MDP can be represent by \\( <S, A, P, R> \\), and our model is going to understand and simulate this. We will only introduce the simple version here, in which we assume that the \\( S \\) and \\( A \\) is known, and thus we only need to model \\( P \\) and \\( R \\). We can formulate it as:

$$
S_{t+1} ~ P_\eta(S_{t+1}|S_t, A_t) \\
R_{t+1} = R_\eta(R_{t+1}|S_t, A_t)
$$

where the prediction of next state is a density estimation problem and the reward is a regression problem.

## Integrated Architecture
In this architecture, we are going to consider two types of experience. **Real experience** which is sampled from the environment, and **Simulated experience** which is sampled from our model. In the past, we only use the real experience to learn value function/policy. Now, we are going to learn our model from real experience, then plan and learn value function/policy from both real and simulated experience. This is thus called integrated architecture (integration of real and fake), the **Dyna Architecture**. Here is an picture to illustrate what the logic flow of Dyna is like.

![Dyna Architecture](/assets/dyna.png)

According to the Dyna architecture, we can design many algorithm, here is an example of **Dyna-Q Algorithm**:
* Initialize \\( Q(s, a) \\) and \\( Model(s, a) \\) for all \\( s \\) and \\( a \\)
* Do forever:
  - \\( S = \\) current (nonterminal) state
  - \\( A = \epsilon - \text{greedy}(S, Q) \\)
  - Execute action \\( A \\); observe result reward \\( R \\), and state \\( S' \\)
  - \\( Q(S, A) = Q(S, A) + \alpha[R + \gamma max_a Q(S', a) - Q(S, A)] \\) (This is using real experience)
  - Update \\( Model(S, A) \\) using \\( R, S' \\)
  - Repeat \\( n \\) times: (This is using simulated experience to learn value function)
    - \\( S = \\) random previously observed state
    - \\( A = \\) random action previously taken in \\( S \\)
    - Sample \\( R, S' \\) from \\( Model(S, A) \\)
    - \\( Q(S, A) = Q(S, A) + \alpha[R + \gamma max_a Q(S', a) - Q(S, A)] \\)

## Monte-Carlo Tree Search
**Monte-Carlo Tree Search** is a very efficient algorithm to plan once we have a model.
* Given a model \\( M_v \\)
* Simulate $K$ episodes from current state $s_t$ using current simulation policy \\( \pi \\)

$$
{s_t, A_t^k, R_{t+1}^k, S_{t+1}^k, ..., S_T^k} ~ M_v, \pi
$$

* Build a search tree containing visited states and actions
* Evaluate state \\( Q(s, a) \\) by mean return of episodes from \\( s, a \\)
* After search is finished, select current (real) action with maximum value in search tree

In MCMT, the simulation policy \\( \pi \\) improves. Each simulation consists of two phases (in-tree, out-of-tree):
* Tree policy (improves): pick action to maximize \\( Q(S, A) \\)
* Default policy (fixed): pick action randomly

Repeat (each simulation):
* Evaluate states \\( Q(S, A) \\) by Mento-Carlo evaluation
* Improve tree policy, e.g. by \\( \epsilon-\text{greedy}(Q) \\)s

There are several advantages of MCMT:
* Highly selective best-first search
* Evaluates states dynamically
* Uses sampling to break curse of dimensionality
* Works for "black-box" models (only requires samples)
* Computationally efficient, anytime  
