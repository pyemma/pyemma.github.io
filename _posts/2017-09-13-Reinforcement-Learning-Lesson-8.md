---
  layout: post
  title: "Reinforcement Learning Lesson 8"
---
This is the last lesson for the entire reinforcement learning, and in this lesson we will learn something related to exploit and explore. In machine learning service, like recommendation service, there is always a trade off between exploit and explore. Exploit means we are always choosing the best given the current information we have, while explore means try something new we haven't tried yet. An example is if you go to restaurant, you can always go to the one you enjoy most(exploit), while you can also try a new one(explore).

This problem is usually formularized as multi bandit problem, which can be represented as $<A, R>$. Here $A$ is a set of action we can take, and $R^a(r) = P[R=r, A=a]$ is an **unknown** probability distribution over rewards. At each time, our agent is going to pick an action, and the environment will generate a reward. The goal is to maximize the cumulative reward.

#### Regret
We can measure the goodness of our action use **regret**. Suppose the action value is the mean reward for an action $a$

$$
Q(a) = E[r|a]
$$

and the optimal value $V^\star$ is the max mean reward we can get

$$
V^\star = Q(a^\star) = max_{a\in A}Q(a)
$$

Then maximize the cumulative reward is equivalent to minimize the total regret, which is

$$
L_t = E[\sum_{i=1}^t (V^\star - Q(a_i))]
$$

#### Upper Confidence Bound
We can try to solve this problem in the face of uncertainty. The best action we should try is the one that would on one hand has a high mean reward, and on the other hand have a high uncertainty. We might get a higher reward, which is good. While we can also get a worse reward, but that does not matter, since we can reduce our uncertainty about that action, and prefer other action which might have higher reward. A more formal description is as follow:
* Estimate an upper confidence $\hat{U_t}(a)$ for each action value, which depends on the number of times $a$ has been selected, the larger the times, the smaller the upper confidence
* Such that $Q(a) \le \hat{Q_t}(a) + \hat{U_t}(a)$ with high probability
* Select action maximize Upper Confidence Bound (UCB)

$$
a_t = argmax_{a\in A} \hat{Q_t}(a) + \hat{U_t}(a)
$$

We need to come up with some method to calculate the upper bound. Here, we bring *Hoeffding's Inequality* for help
> Let $X_1,..., X_t$ be i.i.d. random variables in $[0, 1]$, and let $\bar{X_t} = \frac{1}{i} \sum_{i=1}^t X_i$ be the sample mean. Then

$$
P[E[X] > \bar{X}_t + u] \le e^{-2tu^2}
$$

With this we can have

$$
P[Q(a) > \hat{Q_t}(a) + \hat{U_t}] \le e^{-2N_t(a)U_t(a)^2}
$$

where $N_t(a)$ is the expected number of $a$ is selected. We then can pick a probability $p$ that true value exceeds UCB, and reduce $p$ as we observer more rewards, e.g. $p = t^{-4}$. Then we could obtain the upper bound as:

$$
U_t(a) = \sqrt{\frac{2logt}{N_t(a)}}
$$

And finally we have the UCB1 algorithm

$$
a_t = argmax_{a\in A} (Q(a) + \sqrt{\frac{2logt}{N_t(a)}})
$$
