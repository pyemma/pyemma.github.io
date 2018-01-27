---
  layout: single
  title: DQN In Practice
---
Recently I have been working on Deep-Q-Learning and apply it to some interesting AI games. In this post, I would like to give a brief introduction to how I implemented the Deep-Q-Learning, as well as lots of learning along the way.

### What is DQN
To understand DQN, we need first know is prototype, Q-Leanring. Here is a pervious post about [Q-Learning](https://pyemma.github.io/Reinforcement-Learning-Lesson-4/). Some core elements are:

1. We have a \\( Q(s, a) \\) to record for each state and action pair, what is the expected reward we can get from them
2. We update this estimation by finding what is the **max** reward we can get from the next state leaded by our current state and action, update it by $$Q(S, A) = Q(S, A) + \alpha(R + \gamma max_a Q(S^\prime, a) - Q(S, A))$$

If we have limited number of state and action, we can hold these information into a simple lookup table. However, in reality we usually deal with unlimited number of state and action. In this case, a lookup table is not scalable, we use a model to simulate this part: describe the state with some features, tell the model and the model will tell us what \\( Q(s, a) \\) would be, the model would be trained and updated along the way with the examples we have.

Deep-Q-Leanring basically is a combination of the above two ideas. Apply the logic of Q-Learning, with a model measuring the \\( Q(s, a) \\). Here the *Deep* comes from the fact that we usually use *Deep-Neural-Network* as our model. However, there is another two important thing to stabilize the training of DQN:

1. **Experience Replay**: Instead of directly using the most recent example, we keep a pool of past experience and sample a batch from this pool to update our model
2. **Q-Target Network**: Instead of the max value output by our current model, we use the version of several steps ago. This is called the Q-Target model and this model will be frozen and not updated, but occasionally copied from our main model.

### DQN Implementation
Cool, as we have some highlight idea on what DQN is, let's see how it is implemented. The code is [here](https://github.com/pyemma/tensorflow/blob/master/util/dqn.py). Please not that this code is currently not generalized yet and only suitable for training *Cartpole* game due to how we parsing the state. Making it generalized is WIP. However, that does not prevent us from understanding the main idea of DQN. Let me now illustrate some important component:

Let's first take a look at the main training logic:
{% highlight python %}
def train(self, epsiode=1000):
    """Train the model
    Args:
        epsiode:        Number of epsiode to train
    """
    epsilon = self.epsilon_start

    for ep in range(epsiode):
        state = self.env.reset()
        done = False
        while not done:
            action = self._action(self._norm(state), epsilon)
            next_state, reward, done, _ = self.env.step(action)
            reward = -100 if done else 0.1
            self._remember(state, action, reward, next_state, done)
            state = next_state

        self._learn()

        if (ep+1) % self.step_to_copy_graph == 0:
            self._copy_graph()

        if epsilon > self.epsilon_end:
            epsilon *= self.epsilon_decay
{% endhighlight %}

For each episode, we first initialize the state, and before the game terminate, we take a action based on our model and policy, then get the reward and next state for that action. We then put this as an experience into our memory pool. After the game is terminated, we update our model, and check if we should update q-target network. We also decrease the epsilon as we play. Here we are using \\( \epsilon \\)-greedy policy, and this parameter is the tradeoff between explore and exploit.

Now lets take a look at how we train the model:
{% highlight python %}
def _learn(self):
    """Use Experience Replay and Target Value Network to learn
    """
    if len(self.memory) < self.batch_size:
        return
    sample_idx = np.random.choice(min(len(self.memory), self.memory_size), self.batch_size)
    samples = [self.memory[idx] for idx in sample_idx]

    q_X, target_X, actions, rewards, dones = [], [], [], [], []
    for state, action, reward, next_state, done in samples:
        q_X.append(state)
        target_X.append(next_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    q_labels, target_labels = self.q_model.predict(self.sess, np.array(q_X)), self.target_model.predict(self.sess, np.array(target_X))
    q_target = q_labels.copy()
    q_target[np.arange(self.batch_size), np.array(actions)] = np.array(rewards) + self.gamma * np.max(target_labels, axis=1) * (1 - np.array(dones))

    self.q_model.update(self.sess, np.array(q_X), q_target)
{% endhighlight %}

Here, we sample a batch of experience from our memory pool. Then prepare it into the right format. Our goal is to train our model's prediction (in this case, the prediction is the value of each action) is the same as the actual reward + q-target. In the code, we first get the model prediction for all actions. We also get q-target prediction for each action. We then update the value for the action we take to the target value. Then we train our model using this updated value. Since we only updated the value of action we took, the model will only learn from these updated value, all other is the same as before and model would not learn from them.

### DQN In Practice
During the implementation of this feature, I encountered lots of problem and would like to notice them down for further discussion:
1. Initially I updated the model **after we take each action** instead of **after each game**. This will dramatically increase the number of training we have and impact on the training time. However, getting more number of training is not always a good thing. I noticed that in my case, the training would be not stable.
2. Parameter tuning is really challenging. I tried different combination of batch size, memory pool size, learning rate, and model arch. I found that usually have a moderate memory pool size with a larger learning rate is beneficial.
3. The step to copy the q-target network is also hard to set. If we set is too small, then the training is less stabile; if too large, the training does not get improved.
4. I feel like the usage of the memory is not good enough, as there is not difference in terms of success experience and failure experience. From our common sense, we know that we learn more from our bad experience, maybe we should skew more onto the bad experience? 
