# Intro

This is a simple implementation of Q-learning algorithm. The agent is trying to reach the goal by learning the best action to take in each state. The agent will learn the best action to take in each state by updating the Q-table. The Q-table is a matrix where the rows represent the states and the columns represent the actions. The agent will update the Q-table by using the following formula:

```
Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
```

where:

-   Q(s,a) is the value of the Q-table at state s and action a
-   alpha is the learning rate
-   reward is the reward for taking action a in state s
-   gamma is the discount factor
-   max(Q(s',a')) is the maximum value of the Q-table at state s' and all possible actions a'
-   s' is the next state
-   a' is the next action

# How to run

Before run this. you have to run rl_env for virtual environment.

```bash
source rl_env/bin/activate
```

```bash
python lunar_lander_DQN.py
```

# Results

```bash
python lunar_lander_random.py
```

1. random actions
   Without any reinforcement, random actions will be taken. The agent will not learn anythin.
   Also it is more likely that the agent will not reach the goal.

```bash
python lunaer_lander_Q_learning.py
```

2. Q-learning
   With Q-learning, the agent will learn the best action to take in each state. The agent will update the Q-table by using the formula above. The agent will learn the best action to take in each state by updating the Q-table. The Q-table is a matrix where the rows represent the states and the columns represent the actions. The agent will update the Q-table by using the following formula:

```bash
python lunar_lander_DQN.py
```

3. DQN
   With DQN, the agent will learn the best action to take in each state by using a neural network. The neural network will take the state as input and output the Q-values for each action. The agent will update the neural network by using the following formula:

4. Double DQN
   With Double DQN, the agent will learn the best action to take in each state by using two neural networks. The first neural network will take the state as input and output the Q-values for each action. The second neural network will take the state as input and output the Q-values for each action. The agent will update the first neural network by using the following formula:

```bash
Q(s,a) = Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
```

```bash
python

where:
# Conclusion

With q-learning, it takes forever to reach the goal. With DQN, the agent can reach the goal in a few episodes. DQN is more efficient than q-learning.
```
