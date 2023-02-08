import numpy as np
import heapq
from typing import Union


class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else:
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = None
        n = len(self.adj_mat)

        # Initialize set that stores visited nodes
        visited = set()

        # Setting arbitrary start node
        start = 0
        end = 0

        # Initialize heap and mst
        heap = [(0, (start, end))]
        mst = np.zeros(self.adj_mat.shape)

        # Loop through until visited set contains all nodes
        while len(visited) < n:
            # Select the node in heap with least costly edge
            w, (x, y) = heapq.heappop(heap)

            # If the destination node is not in visited, we can add it to the mst
            if y not in visited:
                mst[x, y] = mst[y, x] = self.adj_mat[x, y]
                visited.add(y)

                # We will also add all neighbors (outgoing edges) to the heap
                # We can find these by looking for edge weights that nonzero
                for v in range(0, n):
                    edge_weight = self.adj_mat[y][v]
                    if edge_weight != 0:
                        heapq.heappush(heap, (edge_weight, (y, v)))

        self.mst = mst

