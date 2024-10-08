import torch
import numpy


class BatchGraphValueIteration():
    def __init__(self, connection_table, reward_function):
        self.connection_table = connection_table
        self.reward_function = reward_function

    def run(self, n_iters=100):
        self.values = torch.zeros_like(self.reward_function)

        for _ in range(n_iters):
            self.values = self.reward_function + torch.max(torch.einsum('...ij,...j->...ij', self.connection_table, self.values), dim=-1).values
        return self.values

    def policy(self, node_idx, values):
        value_map = torch.einsum('...ij,...j->...ij', self.connection_table,  values)
        return torch.argmax(value_map[node_idx], dim=-1)

    def rollout(self, start_node, values, steps=20):
        current_node = start_node
        path = [current_node]
        for t in range(steps):
            current_node = self.policy(current_node, values)
            path.append(current_node)
        return path


if __name__ == '__main__':
    from value_iteration.maze import BatchMaze
    batch_maze = BatchMaze()
    con_batch = batch_maze.get_batch_connections()

    reward = torch.zeros(con_batch.shape[0], con_batch.shape[1])
    for i in range(con_batch.shape[0]):
        ind = numpy.random.randint(0, con_batch.shape[1]-20)
        reward[i, ind] = 1

    vi = BatchGraphValueIteration(con_batch, reward)
    values = vi.run()
    # print(values.shape)
    # paths = vi.rollout(torch.zeros(values.shape[0]), values)

    mazes_v = batch_maze.batch_visualize_energy(values)
    import matplotlib.pyplot as plt
    for maze in mazes_v:
        plt.imshow(maze)
        plt.show()
