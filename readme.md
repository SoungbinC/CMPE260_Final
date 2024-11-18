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
# CMPE260_Final
