# QDVE.DE Predictions using Reinforced Learning with Chainer - Implementation of a Dueling Double DQN

Building a prototype neural network for reinforced learning using chainer, to predict QDVE.DE stock price.

DQN, or Deep Q-Network was the neural network of choice. DQN is a deep neural network that is using reinforcement learning in this case.

The script defines an environment called "Environment1", which represents the financial market, and a Q-Network called "Q_Network", which is the neural network used by the DQN. The DQN is trained on the training data using the Q-learning algorithm with experience replay.

Two very important libraries are Chainer (a deep learning library) and Yahoo Finance (yfinance). Thanks to Chainer we are able to build a strong DQN with reinforced learning and thanks to yfinance we can download the data of the stock prices.

### Understanding the code

The "Environment1" class has a constructor that takes in the historical data and a parameter called "history_t". The "history_t" parameter specifies the number of previous closing prices that the environment keeps in memory. The "reset" method of the class resets the environment to its initial state, and the "step" method takes in an action and returns the next state, reward, and whether the episode is done or not.

The "train_dqn" function takes in an instance of the "Environment1" class. It defines the "Q_Network" class, which is a neural network that takes in the current state (which includes the position value and the previous closing prices) and outputs the Q-values for the available actions (buy, sell, or hold). The "Q_Network" class has three fully connected layers with ReLU activation functions. The function then initializes two instances of the Q_Network class, Q and Q_ast, where Q_ast is a copy of Q.

The function then sets the hyperparameters for training the DQN.

Afterwards it initializes the replay memory, sets the current time step to zero, and starts the training loop. In the training loop, the environment resets, and the agent takes actions based on the epsilon-greedy policy (with Q as the Q-function). The results of the actions are stored in the replay memory. When the replay memory is full, the agent samples batches from the memory and updates the Q_Network using the Q-learning algorithm. Every few batches, the Q_ast network is updated with the Q network's parameters. Finally, the function returns the Q_Network and the training results.

The functons "train_ddqn" and "train_dddqn" follow the same logic as the function "train_dqn" for a double DQN and a dueling double DQN respectively.

### Model Results

DQN had the best result followed by double DQN, with dueling double DQN as the worst performer.


### Final conclusion

Is it true that a simple linear regression has better performance than most of the more complex models? Probably, but I believe it is a matter of time till we optimize and speed up the process of finding out the best performing neural networks for given use case.




