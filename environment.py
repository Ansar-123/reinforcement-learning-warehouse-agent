class WarehouseEnv:

    def __init__(self):

        # size of the warehouse grid
        self.grid_size = 5

        # start position of the robot
        self.start = (0, 0)

        # goal position (pickup / delivery point)
        self.goal = (4, 4)

        # shelves / obstacles inside the warehouse
        self.obstacles = [(1, 1), (2, 2), (3, 1)]

        # current position of the robot
        self.state = self.start


    def reset(self):
        """
        Reset the environment to the starting state
        """
        self.state = self.start
        return self.state


    def step(self, action):
        """
        Perform an action and return next_state, reward, done
        """

        row, col = self.state

        # actions
        if action == 0:      # up
            row -= 1
        elif action == 1:    # down
            row += 1
        elif action == 2:    # left
            col -= 1
        elif action == 3:    # right
            col += 1

        next_state = (row, col)

        # check if out of bounds
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            next_state = self.state
            reward = -5
            done = False
            return next_state, reward, done

        # check obstacle
        if next_state in self.obstacles:
            next_state = self.state
            reward = -10
            done = False
            return next_state, reward, done

        # check goal
        if next_state == self.goal:
            reward = 10
            done = True
            self.state = next_state
            return next_state, reward, done

        # normal movement
        reward = -1
        done = False

        self.state = next_state

        return next_state, reward, done


    def render(self):
        """
        Display the warehouse grid
        """

        for i in range(self.grid_size):
            for j in range(self.grid_size):

                if (i, j) == self.state:
                    print("R", end=" ")

                elif (i, j) == self.goal:
                    print("G", end=" ")

                elif (i, j) in self.obstacles:
                    print("X", end=" ")

                else:
                    print(".", end=" ")

            print()
if __name__ == "__main__":
    env = WarehouseEnv()   # create environment
    
    env.reset()            # start position
    
    env.render()           # display grid