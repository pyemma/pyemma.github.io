---
  layout: single
  title: "Reinforcement Learning Lesson 5"
---
In this post, we are going to look into how can we solve the real world problem with a practical way. Think of the state value function $v(s)$ or the action value function $q(s, a)$ we mentioned before. If the problem has a really large state space, then it would take a lot of memory to store each value function. Instead of recording each value function, we can actually use a model to approximate the actual value function, which means given the current state, we want to predict the value of the state. There are three types of value function approximation:
* Input current state, output the state value
* Input current state and an action, out put the action value
* Input current state, output all possible action's action value

This can be reviewed as a classical supervised learning problem if we **know the actual value function**, and more accurately speaking, its a regression problem. In the regression problem, we are trying to fit a model which will output some real number that matches the our input label as much as possible. In the regression problem, the loss is defined using mean-square error. In order to get a model, we need first to do some feature engineering and represent each state using the **feature vector** $x(S)$, this is going to be the input into our model. And then we try to minimize

$$
L(w) = E_{\pi}[(v_{\pi}(S) - v(S, w))^2]
$$

Here $w$ is our model's parameter and is what we are going to improve. $v_{\pi}(S)$ is the actual value (label) and $v(S, w)$ is the output from our model (predict). In order to minimize this loss, we use stochastic gradient decrease to update $w$, which we have:

$$
\Delta w = \alpha (v_{\pi}(S) - v(S, w)) \nabla_w v(S, w)
$$

Here $\alpha$ is a learning rate controlling how fast we improve $w$, and $\nabla_w v(S, w)$ is the derivate of our model toward the parameter, for example, if we choose a linear model, where $v(S, w) = x(S)^T * w$, then we would have

$$
\Delta w = \alpha (v_{\pi}(S) - v(S, w))x(S)
$$

However, we could only obtain this update when we really **know the actual value function**, which is the case of supervised learning. However, in reinforcement learning, we are lack of such information. So we have to use some target to replacement them. We can actually combining it with the algorithm we have introduced before. For example the MC algorithm, In each episode, we will get a series of the state and corresponding return $<S_t, G_t>$, we can actually use this return as our target and train our model on it. The process would be like use our model to compute the state value, and use some policy to go through the process, then we would have $<S_1, G_1>, <S_2, G_2>, ..., <S_T, G_T>$. Then use these as our training data and update our model. This training is **on-policy** (because we are learning as well as behaving) and **incremental** (episode by episode). Similar things can be applied to TD(0) and TD($\lambda$), where we use TD target and $G_t^\lambda$. Good news to use TD target is that is needs less steps for model to converge (since TD target is less variance), but it might not converge in some cases, for example, if we choose Neural Network as our model, then the model will blow up.

Besides the incremental method, there is also **batch** method, which we record all experience of the agent in $D$, and sample from it to get the training sample, then we update our model parameter using the same method above. **Batch** method is more sample efficient and tries to find the best fit of all value functions available. While in the **incremental** one, we are generate training sample one by one which is not very efficient, and we only use it once after update the parameter. A more detailed example is Deep Q-Networks (DQN), you can think of it as using NN model along with Q learning method. The algorithm is as follow:
* Take action $a_t$ according to $\epsilon$-greedy policy
* Store transition $(s_t, a_t, r_{t+1}, s_{t+1})$ in replay memory $D$
* Sample random mini-batch of transitions $(s, a, r, s^,)$ from $D$
* Compute Q learning target with an old, fixed parameter $w^-$
* Optimize MSE between Q target and Q learning Network

$$
L_i(w_i) = E_{s,a,r,s^, ~ D}[(r + \gamma max_{a^,}Q(s^,,a^,; w^-) - Q(s, a; w_i))^2]
$$

The key method that stabilize the model is experience reply and Q target. For the experience reply, it helps decouple the relationship between each step since we are randomly sampling. For the Q target, we are using the model several steps ago, not the model we just updated. You can think of this as avoid oscillation.
