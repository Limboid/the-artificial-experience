"""Supervised Maximum Reward (SVR) algorithm.

(CoPilot thought up the acronym SVR, but I prefer the full name. 
He wrote most of the code including my opinion on this line.)

The agent is trained to find the optimal policy for the environment by:
- maximizing predicted reward
- minimizing prediction error
- predicting reward
"""