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


# Policy symbols
actions_map = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→"
}

print("Learned Policy Map:\n")

for i in range(env.grid_size):

    for j in range(env.grid_size):

        if (i, j) in env.obstacles:
            print("X", end=" ")

        elif (i, j) == env.goal:
            print("G", end=" ")

        else:
            action = agent.q_table[i, j].argmax()
            print(actions_map[action], end=" ")

    print()