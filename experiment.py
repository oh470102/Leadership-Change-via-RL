from agent import *
from timeit import *
from classes import Env
import matplotlib.pyplot as plt

def resolve_matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
resolve_matplotlib_error()

def running_average(arr):
    cumsum = np.cumsum(arr)
    indices = np.arange(1, len(arr) + 1)
    averages = cumsum / indices
    return averages


'''
Create the cart-pole environment (custom)
    - set render_mode to 'human' for visualization
    - set control_mode to...
        - 'pid1' for SISO (single-input single-output, can only handle the pole's angle)
        - 'pid2' for MIMO (multi-input multi-output, can handle both the pole's angle & position)
'''
env = Env()

'''
Create the reinforcement learning (RL) agent
    - takes the environment as the input parameter, which must be created beforehand
'''
agent = Agent(env=env)

# Load latest model
def t(): 
    scores = agent.train(save=True)

tt = timeit(stmt=lambda: t(), number=1)
print(f"---Training Completed in {tt:2f} seconds---")

# agent.actor.load_state_dict("./saved_models/usethisone.pth")
agent.actor.eval()

## for histogram data
frequency_list = [0 for _ in range(365)]
action_change_list = []

for i in tqdm(range(1000)):
    state = agent.env.reset(for_test=False)
    done, score, prev_action, i = False, 0, None, 0
    action_change = 0

    # episode main loop
    while not done:
        

        # get / process / add noise / clip action
        action = agent.get_action(state, epsilon=False)
        if action != prev_action and prev_action is not None: 
            frequency_list[i] += 1
            action_change += 1
        prev_action = action
        
        # take step in environment
        next_state, reward, done, _ = agent.env.step(action)

        if done: action_change_list.append(action_change)

        # save experience in buffer
        state = next_state

        # record process
        i += 1


print(frequency_list)
categories = [i for i in range(0, 365)]
plt.bar(categories, frequency_list, color='green')


# Customize the plot (optional)
plt.title('Frequency Distribution of Leadership Change')
plt.xlabel('Days Since Project Inception')
plt.ylabel('Frequency')

# Display the plot
plt.savefig('bar.png')
plt.show()

print(f"가장 높은 횟수를 보인 날: {frequency_list.index(max(frequency_list))}")
print(f"평균 리더십 변경 회수: {sum(action_change_list)/len(action_change_list)}")