import numpy as np
import torch

import matplotlib.pyplot as plt


class Maze():
    def __init__(self, size=(10, 10), n_obstacles=10):
        self.maze = self.generate_random_maze(size, n_obstacles)
        self.maze_graph, self.connections = self.from_maze_to_connected_graph(self.maze)

    def generate_random_maze(self, size=(10, 10), n_obstacles=3):
        maze = np.zeros(size)
        # Add obstacles as 2x2 squares
        for _ in range(n_obstacles):
            x, y = np.random.randint(0, size[0] - 1), np.random.randint(0, size[1] - 1)
            maze[x:x + 2, y:y + 2] = 1
        return maze

    def from_maze_to_connected_graph(self, maze):
        # Create a graph with a node for each free cell in the maze.
        # Two nodes are connected if they are adjacent and both are free.

        list_of_free_cells = np.argwhere(maze == 0)
        n_free_cells = len(list_of_free_cells)
        # Build a table with the connections for each cell
        connection_table = np.zeros((n_free_cells, n_free_cells))
        for i, (x1, y1) in enumerate(list_of_free_cells):
            for j, (x2, y2) in enumerate(list_of_free_cells):
                if (np.abs(x1 - x2) + np.abs(y1 - y2)) < 2:
                    connection_table[i, j] = 1
        return list_of_free_cells, connection_table

        print(list_of_free_cells)

    def visualize_energy(self, values):

        maze_rgb = np.stack([self.maze, self.maze, self.maze], axis=-1)
        values = (values - np.min(values))/(np.max(values) - np.min(values))

        for i, (x, y) in enumerate(self.maze_graph):
            maze_rgb[x, y] = [values[i], 0, 1]

        return maze_rgb



class GraphValueIteration():
    def __init__(self, connection_table, reward_function):
        self.connection_table = connection_table
        self.reward_function = reward_function

    def run(self, n_iters=100):
        self.values = np.zeros_like(self.reward_function)

        for _ in range(n_iters):
            self.values = self.reward_function + np.max(np.einsum('ij,j->ij', self.connection_table, self.values), axis=1)

        return self.values



maze = Maze()
connections = maze.connections

reward_function = np.zeros(connections.shape[0])
ind = np.random.randint(0, connections.shape[0])
reward_function[ind] = 1

vi = GraphValueIteration(connections, reward_function)
values = vi.run()

maze_img = maze.visualize_energy(values)
plt.imshow(maze_img)
plt.show()
