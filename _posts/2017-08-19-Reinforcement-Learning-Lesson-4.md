---
  layout: single
  title: "Reinforcement Learning Lesson 4"
---
In this lecture, we learn how to solve an unknown MDP. In the last lecture, we introduced how to calculate the value function given a policy. In this one, we will try to find the optimize policy by ourselves.

#### Mento Calro Policy Iteration
In the [Lesson 2](http://pyemma.github.io/posts/Reinforcement-Learning-Lesson-2), we mentioned how to solve a MDP when we have full information about the MDP. One method is called **Policy Iteration**. It can be divided into two components: *policy iterative evaluation* and *policy improvement*. For the evaluation part, we can use the methods in [last lesson](http://pyemma.github.io/posts/Reinforcement-Learning-Lesson-3), nominally MC and TD. However, we could not directly use the state value function, cause in the policy improvement step (e.g. greedy), we need to know the $R$ and $P$ to find the best action (recall the [Bellman Optimality Function](http://pyemma.github.io/posts/Reinforcement-Learning-Lesson-1)). However, action value function does not need the model of the MDP while in greedy policy improvement:

$$
v_*(s) = argmax_a q(s, a)
$$

For the policy improvement part. If we stick to the greedy method, it will not be good for us to explore all possible states. So we use another method which is called $\epsilon$-greedy. We will have $1-\epsilon$ probability to perform greedily (choose the current best action), and have $\epsilon$ probability to random choose an action:

$$
\pi(s|a) = \begin{cases}
\frac{\epsilon}{m} + 1 - \epsilon, & \text{if $a^\star = argmax_a Q(s, a)$} \\
\frac{\epsilon}{m}, & \text{otherwise}
\end{cases}
$$

We have the final Mento Calro Policy Iteration as:
* Sample the kth episode $S_1, A_1, ..., S_T$ from policy $\pi$
* For each state $S_t$ and $A_t$ in the episode

$$
N(S_t, A_t) = N(S_t, A_t) + 1 \\
Q(S_t, A_t) = Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}((G_t - Q(S_t, A_t))) \\
$$

* Update the $\epsilon$ and policy:

$$
\epsilon = 1/k \\
\pi = \epsilon\text{-greedy}(Q)
$$

#### Sarsa Algorithm
If we use the logic in TD for the evaluation part, then we would have the sarsa algorithm. The main difference is that, in original TD, we use the value state function of the successor state, however, we need the action value function right now. We can obtain that by run our current policy again (remember, TD does not need the complete sequence of experience, we can generate the state and action along the way). Following is the algorithm:
* Initialize $Q$ for each state and action pair arbitrarily, set $Q(terminate, *)$ to 0
* Repeat for each episode
  * Initialize $S$, choose $A$ from the current policy derived from $Q$
  * Repeat for each step in the episode until we hit terminal
    * Take action $A$, observe $R$ and $S^\prime$
    * Choose $A^\prime$ from $S^\prime$ from the current policy derived from $Q$
    * Update $$Q(S, A) = Q(S, A) + \alpha(R + \gamma Q(S^\prime, A^\prime) - Q(S, A))$$
    * Update $S = S^\prime, A = A^\prime$

Similarly, we can also use Eligibility Trace for the sarsa algorithm and result in sarsa($\lambda$) algorithm. The algorithm is as follow:
* Initialize $Q$ for each state and action pair arbitrarily, set $Q(terminate, *)$ to 0
  * Repeat for each episode
  * Initialize $E$ for each $s, a$ pair to 0
  * Initialize $S$, choose $A$ from the current policy derived from Q
  * Repeat for each step in the episode until we hit terminal
    * Take action $A$, observe $R$ and $S^\prime$
    * Choose $A^\prime$ from $S^\prime$ from the current policy derived from Q
    * Calculate $\delta = R + \gamma Q(S^\prime, A^\prime) - Q(S, A)$
    * Update $E(S, A) = E(S, A) + 1$
    * For each $s$ and $a$ pair

$$
Q(s, a) = Q(s, a) + \alpha\delta E(s, a) \\
E(s, a) = \gamma\lambda E(s, a)
$$

* Update $S = S^\prime, A = A^\prime$

#### Q Learning
Both MC policy iteration and sarsa algorithm are **online learning** method, which means that they are observing there own policy, learning along the process. There is another category which is called **offline learning**, in which we learn from other policy, not the policy we are trying to improving. Example is that a robots learns walking by observing human. Q learning falls in this category. It is pretty similar to the sarsa algorithm, the only difference is that when we get the action for successor state, we replace the $\epsilon$-greedy to greedy policy. The Q learning method is as follow:
* Initialize Q for each state and action pair arbitrarily, set Q(terminate, *) to 0
* Repeat for each episode
  * Initialize $S$, choose $A$ from the current policy derived from Q
  * Repeat for each step in the episode until we hit terminal
    * Take action $A$, observe $R$ and $S^\prime$
    * Update $$Q(S, A) = Q(S, A) + \alpha(R + \gamma max_a Q(S^\prime, a) - Q(S, A))$$
    * Update $S = S^\prime$
