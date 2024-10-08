import torch
import numpy as np


class Maze():
    def __init__(self, size=(10, 10), n_obstacles=6):
        self.maze = self.generate_random_maze(size, n_obstacles)
        self.maze_graph, self.connections = self.from_maze_to_connected_graph(self.maze)
        self.n_nodes = len(self.maze_graph)

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

        values = values[:self.n_nodes]
        values = (values - np.min(values))/(np.max(values) - np.min(values))

        for i, (x, y) in enumerate(self.maze_graph):
            maze_rgb[x, y] = [values[i], 0, 0]

        return maze_rgb

    def visualize_path(self, path):
        maze_rgb = np.stack([self.maze, self.maze, self.maze], axis=-1)

        path_lenght = len(path)
        for i in range(path_lenght):
            node = path[i]
            x, y = self.maze_graph[node]

            c = (i/path_lenght)/2 + 0.7
            maze_rgb[x, y] = [0, c, 0]

        return maze_rgb


class BatchMaze():
    def __init__(self, batch_size=32, size=(10, 10), n_obstacles=6):
        self.batch_size = batch_size
        self.mazes = [Maze(size, n_obstacles) for _ in range(batch_size)]

    def get_batch_connections(self):
        connections = [m.connections for m in self.mazes]
        # Each maze has a different number of nodes, so we need to pad the connections
        max_nodes = max([c.shape[0] for c in connections])
        connections_pad =  torch.zeros(self.batch_size, max_nodes, max_nodes)
        for i in range(len(connections)):
            n_nodes = connections[i].shape[0]
            connections_pad[i, :n_nodes, :n_nodes] = torch.tensor(connections[i]).float()
        return  connections_pad

    def batch_visualize_energy(self, values):
        values_np = values.detach().numpy()
        return [m.visualize_energy(v) for m, v in zip(self.mazes, values_np)]


if __name__ == '__main__':

    batch_maze = BatchMaze()
    con_batch = batch_maze.get_batch_connections()
    print(con_batch.shape)