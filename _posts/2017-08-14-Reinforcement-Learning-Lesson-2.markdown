---
layout: single
title: "Reinforcement Learning Lesson 2"
---
In the last post, we introduced the definition of Markov Decision Process and Bellman Equation. Now, if you are given the states $S$, action $A$, transition matrix $P$, rewards $R$ and discounting ratio $\gamma$, how would you come up with a solution for this MDP? i.e. how would you calculate the value function and come up with an optimal policy for it?

#### Value Iteration
This first method is to apply the Bellman Optimality Equation repeatedly. The idea is that we continue update the best estimation for each state value function, and once all $s^\prime$ reachable from $s$ achieve its optimal value function, then $v(s)$ can also achieve the optimal value. The algorithm is as follow:
* Initiate $v(s)$ to 0 for all $s\in S$
* Apply
$$v(s) = max_a(R_s^a + \gamma\sum_{s^\prime\in S}P_{ss^\prime}^av(s))$$
to update each state value function to a better estimation

Why this algorithm guarantee to find the optimal state value function (thus optimal policy)? Its because the Bellman Optimality Equation can be regarded as a contraction. We can image in a value function space, where its dimension is $|S|$, each point in this space determine a value state function. A contraction is an operation that can make two points in this space closer.
> (Contraction Mapping Theory) For any metric space that is complete under an operator that is a contraction, the operator will converge to a unique fixed point.

According to the Contraction Mapping Theory, we know that $v^\ast=Tv^\ast$ has a unique solution. And based on value iteration converge, we know that $v_{t}=Tv_{t-1}$. Then we could have

$$
||v_t - v\ast||_\infty = ||Tv_{t-1} - Tv\ast||_\infty \le \gamma ||v_{t-1} - v\ast||_\infty
$$

By applying the operator repeatedly, we are bringing our estimated value function closer and closer to the fixed point, thus we are achieving the optimal value function gradually.

To prove Bellman Optimality Operator is a contraction, we can have:

$$
\begin{align}
|Tv_1(s) - Tv_2(s)|
& = |max_a(R_s^a + \gamma\sum_{s^\prime\in S}P_{ss^\prime}^av_1(s)) - max_a(R_s^a + \gamma\sum_{s^\prime\in S}P_{ss^\prime}^av_2(s))| \\
& \le max_a|(R_s^a + \gamma\sum_{s^\prime\in S}P_{ss^\prime}^av_1(s)) - (R_s^a + \gamma\sum_{s^\prime\in S}P_{ss^\prime}^av_2(s))| \\
& = max_a|\gamma\sum_{s^\prime\in S}P_{ss^\prime}^av_1(s) - \gamma\sum_{s^\prime\in S}P_{ss^\prime}^av_2(s)| \\
& \le \gamma max_s|v_1(s) - v_2(s)|\\
\end{align}
$$

#### Policy Iteration
Compared with value iteration which focus on computing the optimal value function. Policy iteration evaluate a policy and improve the policy gradually, and finally converge to the optimal policy, the algorithm is as follow:
* Initialize a random policy $\pi$
* Apply Bellman Expectation Equation to all state $s$ to get the current value function $v^\pi$
* Improve the current policy greedily by:

$$\pi^\prime = argmax_a (R_s^a + \gamma \sum_{s^\prime\in S}P_{ss^\prime}^av^\pi(s))$$

* Repeat until the policy does not change

Why policy iteration guarantee to converge to optimal policy? First, we can also proof that the Bellman Expectation Operator(Equation) is also a contraction. Thus given a policy $\pi$, we know that the value function will converge to $v^\pi$. Then, we only need to prove the policy can be improved by our greedy selection.

Suppose a deterministic policy $a = \pi(s)$. We can improve it by acting greedily by

$$\pi^\prime(s) = argmax_aq_\pi(s, a)$$

according to the current action value function(remember the relationship between action value function and state value function, they can be transformed each other). It can improve the value function for any state

$$q_\pi(s, \pi^\prime(s)) = argmax_a q_\pi(s, a) \ge q_\pi(s, \pi(s)) = v_\pi(s)$$

And thus we can improves the value function $v_{\pi^\prime}(s) \ge v_\pi(s)$ (this can be proved by expanding the return and recursively substitute the above function).

Policy iteration is pretty similar to Expectation Maximization (EM). In EM, we first evaluate the data using the current parameters, and then update the parameters to maximize the quantity.

More detailed proof is available [here](http://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/slides/mdps.pdf)
