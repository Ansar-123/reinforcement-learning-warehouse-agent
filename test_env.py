from environment import WarehouseEnv

# create environment
env = WarehouseEnv()

# reset environment
state = env.reset()

print("Initial State:", state)

# show grid
env.render()

print()

# test some actions
actions = [3, 3, 1, 1, 1]  

for action in actions:

    next_state, reward, done = env.step(action)

    print("Action:", action)
    print("Next State:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    env.render()

    print()

    if done:
        print("Goal reached!")
        break