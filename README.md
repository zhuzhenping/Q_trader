# Q_trader
A reinforcement learning based auto-trader in stock market

This is a project base on online course Machine Learning for Trader. The testcase and the framework are obtained online. All other codes are written by Y. Jiao (yjiao03@syr.edu).

The codes utilize naive Q-learning algorithm. No machine learning package is used, and all methods were written by the author. Four technical indicators are calculated (in Figure). The learner is trained by one-year those indicators, and can be tested by other data. The best strategy yields better average return than benchmark trading strategy in the tested data range, but no better on the training dataset. An improved version, deep Q-learning has been developed by the author in DeepQ folder, which combined artifical neural network and Q-learning. But the codes are not fully tested.

run

python teststrategy.py

Need to modify the path before run the program.
