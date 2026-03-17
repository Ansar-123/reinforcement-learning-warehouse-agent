from environment import WarehouseEnv
from agent import QLearningAgent

# create environment
env = WarehouseEnv()

# create agent
agent = QLearningAgent(env.grid_size)

# train the agent first
episodes = 2000

for episode in range(episodes):

    state = env.reset()
    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update_q(state, action, reward, next_state)

        state = next_state

    agent.decay_epsilon()

print("Training completed!\n")


# Now run the trained agent
state = env.reset()
done = False

print("Robot following learned policy:\n")

env.render()
print()

steps = 0

while not done and steps < 50:

    row, col = state

    # choose best action from Q-table
    action = agent.q_table[row, col].argmax()

    next_state, reward, done = env.step(action)

    state = next_state

    env.render()
    print()

    steps += 1

if done:
    print("Goal reached successfully!")
else:
    print("Agent stopped.")