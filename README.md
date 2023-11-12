## Leadership-Change-via-RL
Analyzing the ideal timing of switching leaders in large projects via RL approaches

## Markov Decision Process (MDP) 
The MDP space established for this research is as follows.
$$\mathcal{M} = <\mathcal{S}, \mathcal{A}, \mathcal{R}, \gamma>$$

- $\mathcal{S}$ describes the available state spaces, which in this case is a tuple of the following form: (current leader index, $i$, resource, HP). This is essentially what the agent _observes_ before making decisions.
- $\mathcal{A}$ is the available set of actions, which $a \in (0, 1, ..., 9)$. $a$ here represents the new leader's index. 
- $\mathcal{R}$ is a single element, equal to $R$ drawn from the normal distribution. $gamma$ was set as 0.95 as a constant.


