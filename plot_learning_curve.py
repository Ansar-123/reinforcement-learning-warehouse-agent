import matplotlib.pyplot as plt
from environment import WarehouseEnv
from agent import QLearningAgent

episodes = 2000

env = WarehouseEnv()
agent = QLearningAgent(env.grid_size)

rewards_per_episode = []

for episode in range(episodes):

    state = env.reset()
    done = False
    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update_q(state, action, reward, next_state)

        state = next_state

        total_reward += reward

    agent.decay_epsilon()

    rewards_per_episode.append(total_reward)

print("Training finished!")

# plot learning curve
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve of Warehouse RL Agent")
plt.show()