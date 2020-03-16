import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_zachary(filename):
    with open("data/" + filename, "r") as f:
        edges = []
        for line in f:
            edge = list(map(int, line.split()))[:2]
            edges.append(edge)

    # add on reversed edges
    rev_edges = [[t, s] for s, t in edges]
    edges = edges + rev_edges

    edge_dict = {}
    for edge in edges:
        s, t = edge
        # edge_dict is of the form
        #   {<edge source>: [list of <edge target>]}
        if s in edge_dict and t not in edge_dict[s]:
            edge_dict[s].append(t)
        elif s not in edge_dict:
            edge_dict[s] = [t]

    return edge_dict


def generate_random_walks(edge_dict, walk_length=5):
    n_vertices = len(edge_dict.keys())
    walks = []
    print("generating walks...")
    for v in tqdm(range(1, n_vertices + 1)):
        walk = [v]
        for _ in range(walk_length):
            last_v = walk[-1]
            next_v = np.random.choice(edge_dict[last_v])
            walk.append(next_v)
        walks.append(walk)

    return np.array(walks)


if __name__ == "__main__":
    edges = parse_zachary("zachary.txt")
    walks = generate_random_walks(edges, walk_length=10)
    print(walks)
