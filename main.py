import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

edges = [
    (0, 1),
    (1, 5),
    (5, 6),
    (5, 4),
    (1, 2),
    (1, 3),
    (9, 10),
    (2, 4),
    (0, 6),
    (6, 7),
    (8, 9),
    (7, 8),
    (1, 7),
    (3, 9),
]

goal = 10

G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)

plt.show()

# Rewarding system
MATRIX_SIZE = 11
M = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
M *= -1

for point in edges:
    print(point)

    if point[1] == goal:
        M[point] = 100
    else:
        M[point] = 0

    if point[0] == goal:
        M[point[::-1]] = 100
    else:
        # reverse of point
        M[point[::-1]] = 0

# add goal point round trip
M[goal, goal] = 100
print(M)

# 4). Utility functions
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# learning parameter
gamma = 0.75
initial_state = 1


def available_actions(state):
    current_state_row = M[state,]
    available_actions = np.where(current_state_row >= 0)[1]
    return available_actions


available_action = available_actions(initial_state)
print(available_action)


def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range, 1))
    return next_action


action = sample_next_action(available_action)


def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    max_value = Q[action, max_index]
    Q[current_state, action] = M[current_state, action] + gamma * max_value


# Updating the Q-Matrix according to the path chosen
update(initial_state, action, gamma)


# 5). Training
scores = []
for i in range(1000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_action = available_actions(current_state)
    action = sample_next_action(available_action)
    update(current_state, action, gamma)
    scores.append(sum(Q[current_state,]))


print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# 6). Testing
current_state = 0
steps = [current_state]

while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index[0])

    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)


plt.plot(np.arange(len(scores)), np.squeeze(scores))
plt.xlabel("No of iterations")
plt.ylabel("Reward gained")
plt.show()
