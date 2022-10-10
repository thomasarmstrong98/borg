from multiprocessing import Pool
from typing import Callable, Iterable, List, Tuple
from time import time
from itertools import repeat

import networkx as nx
import numpy as np


# default experiment settings
NUM_PROCESSES = 5
NUM_GRAPH_SIMULATIONS = 1
GRAPH_N_RANGE = [10**i for i in range(1, 4)]


def _apply_counter_to_simulations(
    triangle_counter: Callable[[nx.Graph], int],
    graph_size: int = 100,
    num_simulations: int = NUM_GRAPH_SIMULATIONS,
    prob_of_edge_creation: float = 0.25,
) -> Tuple[int, float]:
    """
    Generate a series of random graphs for a specific size and apply the
    triangle counter to each. Returns the graph size and the average
     runtime of the counting algorithm.
    """

    runtimes = []
    for _ in range(num_simulations):
        _s = time()
        graph = nx.fast_gnp_random_graph(
            graph_size, prob_of_edge_creation, directed=False
        )
        triangle_counter(graph)
        _e = time()
        runtimes.append(_e - _s)

    return graph_size, np.mean(runtimes)


def run_experiment(
    triangle_counter: Callable[[nx.Graph], int],
    graph_sizes: Iterable[int] = GRAPH_N_RANGE,
    num_simulations_per_size: int = NUM_GRAPH_SIMULATIONS,
    prob_of_edge_creation: float = 0.25,
) -> List[Tuple[int, float]]:
    """
    Runs the triangle counter on many simulations of graphs of varying size
    """

    with Pool(NUM_PROCESSES) as pool:
        results = pool.starmap(
            _apply_counter_to_simulations,
            zip(
                repeat(triangle_counter),
                graph_sizes,
                repeat(num_simulations_per_size),
                repeat(prob_of_edge_creation),
            ),
        )

    return results
