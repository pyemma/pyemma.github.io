---
  layout: single
  title: "Reinforcement Learning Lesson 6"
  tags:
    - reinforcement learning
---
In the pervious we use a model to approximate the state value/action value function. In this post, we are going to learn how to directly parameterize a policy, which means we would directly get the probability of each action given a state:

$$
\pi_{\theta}(s ,a) = P[a|s, \theta]
$$

In this case, we are not going to have any value function. A slight variance of this method is called **Actor-Critic**, in which both value function and policy are modeled and learnt.

The advantage of **Policy based RL** is:
* Better convergence properties
* Effective in high-dimensional or continuous action spaces
* Can learn stochastic policies

#### Policy Objective Functions
Since we are going to learn \\( \pi_\theta (s, a) \\) and find the best \\( \theta \\), we need to first find a way to measure the quality of our policy. These are called **policy objective function** and some we can use are:
* In episode environment we can use the start value

$$
J_1(\theta) = V^{\pi_\theta}(s_1) = E_{\pi_\theta}[v_1]
$$

* In continuous environment we can use average value or average reward pre time-step

$$
J_{avV}(\theta) = \sum_{s}d^{\pi_\theta}(s)V^{\pi_\theta}(s) \\
J_{avR}(\theta) = \sum_{s}d^{\pi_\theta}(s)\sum_{a}\pi_\theta(s, a)R_s^a
$$

where \\( d^{\pi_\theta}(s) \\) is stationary distribution of Markov chain for \\( \pi_\theta \\).

After we have the measurement of the policy quality, we are going to find the best parameter which gives us the best quality and this becomes an optimization problem. Actually, similar to the last post, we can also use stochastic gradient to help use here. Since we are trying to find the maximum value, we are going to use what is called gradient ascent to find the steepest direction to update our parameter (very similar to gradient decrease).

#### Score Function
In order to compute the policy gradient analytically, we introduced the **score function**. Assume policy $\pi_{\theta}$ is differentiable whenever it is non-zero and we know the gradient $\nabla_\theta \pi_\theta (s, a)$. Then using some tricky we have:

$$
\begin{align}
\nabla_\theta \pi_\theta (s, a) & = \pi_\theta (s, a) \frac{\nabla_\theta \pi_\theta (s, a) }{\pi_\theta (s, a)} \\
& = \pi_\theta (s, a) \nabla_\theta log \pi_\theta (s, a)
\end{align}
$$

Here, $\nabla_\theta log \pi_\theta (s, a)$ is the **score function**.

#### Policy Gradient Theorem
> For any differentiable policy $\pi_\theta (s, a)$, for any of the policy objective functions $J_1, J_{avV}, J_{avR}$, the policy gradient is

$$
\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log \pi_\theta (s, a) Q^{\pi_\theta} (s, a)]
$$

#### Monte-Carlo Policy Gradient
Use return as an unbiased sample of $Q^{\pi_\theta} (s, a)$, the algorithm is as follow:
* Initialize $\theta$ arbitrarily
  - for each episode ${s_1, a_1, r_2, ..., s_{T-1}, a_{T-1}, r_T} ~ \pi_\theta$ do
    - for $t = 1$ to $T - 1$ do
      - $\theta = \theta + \alpha \nabla_\theta log \pi_\theta (s, a) v_t$
    - end for
  - end for
* return $\theta$

#### Actor Critic Policy Gradient
The problem with Monte-Carlo Policy Gradient is that is has a very high variance. In order to reduce the variance, we can use a **critic** to estimate the action value function. Thus in **Actor Critic Policy Gradient**, we have two components:
* *Critic* updates action value function parameters $w$
* *Actor* updates policy parameters $\theta$, in direction suggested by critic

Here is an example when we use linear value function approximation for the critic:
* Initialize $s$, $\theta$
* Sample $a ~ \pi_\theta$
* for each step do
  - Sample reward $r$, sample next state $s'$
  - Sample action $a' ~ \pi_\theta (s', a')$
  - $\delta = r + \gamma Q_w(s', a') - Q_w(s, a)$ (This is the TD error)
  - $\theta = \theta + \alpha \nabla_\theta log \pi_\theta (s, a) Q_w(s, a)$ (We replace with the approximation)
  - $w = w + \beta \delta \phi(s, a)$ (Update value function approximation model parameter)
  - $a = a'$, $s = s'$
* end for
