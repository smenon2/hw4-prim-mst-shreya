import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray,
              mst: np.ndarray,
              expected_weight: int,
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i + 1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    # MST should have n-1 edges
    assert len(np.nonzero(mst)[1]) / 2 == (len(adj_mat) - 1), 'Proposed MST has incorrect number of edges'
    # MST should be symmetric
    assert np.allclose(mst, mst.T, rtol=1e-05, atol=1e-08), 'Proposed MST is not symmetric'

    assert is_connected(mst), 'Proposed MST is not connected'


# This is a helper function that uses BFS to see if a graph is connected
# It will return true if all nodes are visited with BFS from starting node
def is_connected(adj_matrix):
    visited = [False for i in range(len(adj_matrix))]
    queue = []

    start = 0  # arbitrary starting node
    queue.append(start)
    visited[start] = True

    while queue:
        node = queue.pop(0)
        for i, connected in enumerate(adj_matrix[node]):
            if connected and not visited[i]:
                queue.append(i)
                visited[i] = True

    return all(visited)


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path)  # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords)  # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    # We will create our own adjacency matrix and
    adj_matrix = np.array([
        [0, 2, 0, 6, 0],
        [2, 0, 3, 8, 5],
        [0, 3, 0, 0, 7],
        [6, 8, 0, 0, 9],
        [0, 5, 7, 9, 0]
    ])
    g = Graph(adj_matrix)
    g.construct_mst()

    count = 0
    for i in range(len(g.mst)):
        if g.mst[i] is not np.zeros(len(g.adj_mat)):
            count += 1

    # We know that the MST of this tree must be 16
    # Calculate the cost of the proposed MST
    total = 0
    for i in range(g.mst.shape[0]):
        for j in range(i + 1):
            total += g.mst[i, j]

    assert count == len(g.mst), 'MST does not span all nodes'
    assert total == 16, "MST does not have the right cost"
