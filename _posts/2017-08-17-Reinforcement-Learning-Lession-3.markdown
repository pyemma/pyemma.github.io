---
  layout: single
  title: "Reinforcement Learning Lesson 3"
---
In this lesson, we will learn about what to do when we have no knowledge about the MDP. In the last lesson, we learnt about how to solve a MDP when we have full information about it (e.g. $P$, $R$). When we don't have enough information, the Bellman Equation won't work. The only way is to learn from experience, where we run the process once, and obtain a $S_1, R_1, ..., S_T$ sequence and improve our value function with it. This is called model free. In this lesson, we learn about when given a policy $\pi$, how do we calculate the state value function (which is called model free predicting). And in the next one, we will learn how to come up with the policy (which is called model free control).

#### Monte-Carlo Reinforcement Learning
The first method is called Mento-Carlo Reinforcement Learning. The idea behind this method is to use empirical mean to measure the value. The algorithm is as follow:
* Initialize $N(s)$ to all zero, copying the value function from last one
* Given an episode $S_1, R_1, ..., S_T$
* For each $S_t$ with return $G_t$

$$N(S_t) = N(S_t) + 1$$

$$V(S_t) = V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$$

Here $N(S_t)$ counts the number of our visit to $S_t$. The update function is using running mean to update the value of the current state, by moving it towards the return in this episode (G_t) a little bit. Here we can replace $\frac{1}{N(S_t)}$ to a small number $\alpha$, this is functioning as a learning rate to control how quick we update our value function. When we increase the counter, we can increase it either by first visit within the episode or every visit within the episode.

Mento-Carlo Reinforcement Learning can only works with episode experience, which means the MDP must has a terminate state and the experience must be complete.

In this method, we are **sampling** from the policy distribution because for each state, we are only considering one possible successor state. The learning method in last lesson is using dynamic programming, and it is not based on sampling, it actually takes all possible successor states into consideration.

#### TD(0) Learning
The second method is called Temporal Difference Learning. As its naming suggested, in this method we are not using the actual return in the episode but using an temporal estimation to update the value function. The algorithm is:
* For each $S_t$ within the episode

$$
V(S_t) = V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
$$

Here, $R_{t+1} + \gamma V(S_{t+1})$ is called TD target, and $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called the TD error. The main logic here is bootstrapping, which means we are not directly making each value function to the most accurate value it should be given this episode. We are making it slightly better based on our current estimate on the successor state. The benefit of the doing so is that we can learn from incomplete experience, and MDP without a terminal state.

In this method, we are also **sampling** from the policy distribution, as well as bootstrapping. Dynamic programming also uses bootstrapping similar to TD(0) learning (recall the Bellman Equation).

#### TD($\lambda$) Learning
In both MC and TD(0) Learning, we are looking forward to the future rewards. In MC, we are looking until we reach the end, while in TD(0) we only look at next step. Instead of looking forward, we can also looking backward. However, this involves how to assign the current timestamp rewards to pervious states. This is called credit assignment problem. And the method we overcome it is to use **Eligibility Traces**, which fusion both assigning credit to the most recent state and most frequent states. Here we introduce the TD($\lambda$) algorithm (back view version):
* Initialize Eligibility Traces $E_0(s) = 0$
* Given an experience, for each state $s$:
* Update the Eligibility Traces by: $E_t(s) = \gamma \lambda E_{t-1}(s) + 1(S_t = s)$
* Calculate the update step by: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
* Update **each** state by: $V(s) = V(s) + \alpha \delta_t E_t(s)$

If we use $\lambda = 0$, then the Eligibility Traces will fall to $1(S_t = s)$ and replace it in the update function, we will see that its the exact same update function as TD(0). If we choose $\lambda = 1$, then it is actually equals to every visit MC. We can prove it as follow:
* Suppose in our experience, $s$ is visited at timestamp $k$, then the $E_t(s)$ will be like

$$
E_t(s) = \begin{cases}
0, & \text{if $t < k$} \\
\gamma^{t - k}, & \text{if $t \ge k$} \\
\end{cases}
$$

* The accumulated online update for $s$ is

$$
\begin{align}
\sum_{t=1}^{T-1}\alpha\delta_t E_t(s) & = \sum_{t=k}^{T-1}\gamma^{t-k}\delta_t \\
& = \delta_k + \gamma\delta_{k+1} + ... + \gamma^{T-1-k}\delta_{T-1} \\
& = R_{k+1} + \gamma V(S_{k+1}) - V(S_k) + \gamma R_{k+2} + \gamma^2 V(S_{k+2}) - \gamma V(S_{k+1}) + ... \\
& + \gamma^{T-1-k} R_{T-1} + \gamma^{T-k} V(S_T) - \gamma^{T-1-k} V(S_{T-1}) \\
& = R_{k+1} + \gamma R_{k+2} + \gamma^2 R_{k+3} + ... + \gamma^{T-1-k} R_{T-1} - V(S_k) \\
& = G_k - V(S_k)
\end{align}
$$

* Thus the update function for TD(1) is the same as the one in every visit MC (where we use $\alpha$ as a learning rate instead of the original one).

The good thing for TD(\$lambda$) is that it can learn with incomplete experience. And the update is performed *online*, *step by step* within the episode. MC is updated via offline, cause it needs to wait until the end and calculate the update for each state and update them in batch.
