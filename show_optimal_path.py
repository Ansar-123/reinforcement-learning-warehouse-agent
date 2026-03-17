from environment import WarehouseEnv
from agent import QLearningAgent

env = WarehouseEnv()
agent = QLearningAgent(env.grid_size)

episodes = 2000

# Train the agent
for episode in range(episodes):

    state = env.reset()
    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update_q(state, action, reward, next_state)

        state = next_state

    agent.decay_epsilon()

print("Training finished\n")

# Find optimal path
state = env.reset()
path = [state]
done = False

while not done:

    row, col = state

    action = agent.q_table[row, col].argmax()

    next_state, reward, done = env.step(action)

    path.append(next_state)

    state = next_state

# Show the optimal path
print("Optimal Path:")
print(path)