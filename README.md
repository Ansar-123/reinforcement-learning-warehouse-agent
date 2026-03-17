# Warehouse Robot Navigation using Q-Learning

This project is a simple implementation of Reinforcement Learning using the Q-learning algorithm. 
The goal of the project is to train a warehouse robot to move inside a grid environment and reach 
a target location while avoiding obstacles.

I created a custom warehouse environment and trained a Q-learning agent to learn the optimal path 
from the starting position to the goal. The agent improves its decisions over multiple training 
episodes by updating the Q-table based on rewards received from the environment.

## Project Features

- Custom warehouse grid environment
- Obstacles inside the warehouse
- Q-learning based reinforcement learning agent
- Training over multiple episodes
- Visualization of the learned optimal path
- Display of learned policy
- Learning curve visualization

## Technologies Used

- Python
- NumPy
- Matplotlib
- Reinforcement Learning (Q-learning)

## Project Structure

agent.py  
Contains the implementation of the Q-learning agent and Q-table update logic.

environment.py  
Defines the warehouse environment including the grid, obstacles, rewards, and robot movement.

train.py  
Used to train the reinforcement learning agent.

run_trained_agent.py  
Runs the trained agent and shows how it navigates the warehouse.

show_optimal_path.py  
Displays the optimal path learned by the agent.

show_policy.py  
Shows the learned movement policy for each grid cell.

plot_learning_curve.py  
Plots the learning curve based on rewards collected during training.

test_env.py  
Simple script to test the environment behavior.

## How to Run the Project

1. Train the agent

python train.py

2. Run the trained agent

python run_trained_agent.py

3. Show the optimal path

python show_optimal_path.py

4. Display the learned policy

python show_policy.py

5. Plot the learning curve

python plot_learning_curve.py

## Author

Muhammed Ansar  
Computer Science Engineering Graduate  
Interested in Machine Learning and Artificial Intelligence
