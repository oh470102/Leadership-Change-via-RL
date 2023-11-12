## Leadership-Change-via-RL
Analyzing the ideal timing of switching leaders in large projects via RL approaches

## Markov Decision Process (MDP) 
실제로 리더의 변경을 판단하는 것은 강화학습 알고리즘으로 학습된 에이전트(Agent)이다. 본 연구에서 $\mathcal{S}$는 (현재 리더, $i$, Resource, HP)와 같은 튜플, $\mathcal{A}$는 새로운 리더의 인덱스 ($[0, n)$), 그리고 $\mathcal{R}$은 \textbf{식 1}에서 계산한 값을 사용하였다. 학습 과정에서는 [1]에서 제시한 D3QN 알고리즘을 사용하였다.
