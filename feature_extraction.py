import networkx as nx
from math import sqrt


def number_of_triangles(subgraph_obj, is_directed):
    """
    Function to get number of triangles in a subgraph

    @itype  subgraph_obj: Digraph object
    @itype  is_directed: Bool

    @rtype:   int
    """
    adj_mat = nx.adjacency_matrix(
        subgraph_obj, nodelist=None, dtype=None, weight="weight"
    )
    adj_array = adj_mat.toarray()
    nodes = len(adj_array)
    count_triangle = 0

    # Consider every possible
    # triplet of edges in graph
    for i in range(nodes):
        for j in range(nodes):
            for k in range(nodes):

                # check the triplet
                # if it satisfies the condition
                if (
                    i != j
                    and i != k
                    and j != k
                    and adj_array[i][j]
                    and adj_array[j][k]
                    and adj_array[k][i]
                ):
                    count_triangle += 1

    # If graph is directed , division is done by 3
    # else division by 6 is done
    if is_directed:
        return count_triangle // 3
    else:
        return count_triangle // 6


def nodes_and_edges(subgraph_obj):
    """
    Function to get the number of nodes and edges of the subgraph

    @itype  subgraph_obj: Digraph object

    @rtype: tuple (nodes, edges)
    """
    adj_mat = nx.adjacency_matrix(
        subgraph_obj, nodelist=None, dtype=None, weight="weight"
    )
    adj_array = adj_mat.toarray()
    nodes = adj_array.shape[0]
    edges = adj_mat.shape[0] + 1
    return nodes, edges


def katz_centrality(
    graph,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    weight="weight",
):
    """
    Function for analytical solution of Katz Centrality for the subgraph

    @itype  graph: nxGraph
    @itype  alpha: float
    @itype  beta: scalar or dict
    @itype  max_iter: int
    @itype  tol: float
    @itype  nstart: dict
    @itype  normalized: bool
    @itype  weight: str

    @rtype:   dict
    """

    if len(graph) == 0:
        return {}

    nnodes = graph.number_of_nodes()

    if nstart is None:

        x_nodes = dict([(n, 0) for n in graph])

    else:  # if nstart is not None
        x_nodes = nstart

    try:
        b_beta = dict.fromkeys(graph, float(beta))
    except (TypeError, ValueError, AttributeError):
        b_beta = beta
        if set(beta) != set(graph):  # raise size mismatch error
            raise nx.NetworkXError("beta dictionary must have a value for every node")

    for i in range(max_iter):
        xlast = x_nodes
        x_nodes = dict.fromkeys(xlast, 0)

        for j in x_nodes:
            for nbr in graph[j]:
                x_nodes[nbr] += xlast[j] * graph[j][nbr].get(weight, 1)
        for j in x_nodes:
            x_nodes[j] = alpha * x_nodes[j] + b_beta[j]

        err = sum([abs(x_nodes[j] - xlast[j]) for j in x_nodes])
        if err < nnodes * tol:
            if normalized:

                try:
                    std = 1.0 / sqrt(sum(v**2 for v in x_nodes.values()))

                except ZeroDivisionError:
                    std = 1.0
            else:
                std = 1
            for j in x_nodes:
                x_nodes[j] *= std
            return x_nodes

    raise nx.NetworkXError(
        "Power iteration failed to converge in " "%d iterations." % max_iter
    )


def clustering(graph):
    """
    Function to return the clustering inxex for nodes

    @itype  graph: nxGraph

    @rtype: dict
    """
    return nx.clustering(graph)


def k_components(graph):
    """
    Function to return the k component of the subgraph

    @itype  graph: nxGraph

    @rtype: dict
    """
    return nx.k_components(graph)


def eigenvector_centrality(
    graph, max_iter=100, tol=1.0e-6, nstart=None, weight="weight"
):

    """
    Function for analytical solution of eigenvector Centrality for the subgraph

    @itype  graph: nxGraph
    @itype  iter: int
    @itype  tol: float
    @itype  nstart: dict
    @itype  weight: str

    @rtype:   dict
    """

    if type(graph) == nx.MultiGraph or type(graph) == nx.MultiDiGraph:
        raise nx.NetworkXException("Not defined for multigraphs.")

    if len(graph) == 0:
        raise nx.NetworkXException("Empty graph.")

    if nstart is None:

        x_nodes = dict([(n, 1.0 / len(graph)) for n in graph])
    else:
        x_nodes = nstart

    s_rec = 1.0 / sum(x_nodes.values())
    for k in x_nodes:
        x_nodes[k] *= s_rec
    nnodes = graph.number_of_nodes()

    for i in range(max_iter):
        xlast = x_nodes
        x_nodes = dict.fromkeys(xlast, 0)

        for j in x_nodes:
            for nbr in graph[j]:
                x_nodes[nbr] += xlast[j] * graph[j][nbr].get(weight, 1)

        try:
            s_rec = 1.0 / sqrt(sum(v**2 for v in x_nodes.values()))

        except ZeroDivisionError:
            s_rec = 1.0
        for i in x_nodes:
            x_nodes[i] *= s_rec

        err = sum([abs(x_nodes[n] - xlast[n]) for n in x_nodes])
        if err < nnodes * tol:
            return x_nodes

    raise nx.NetworkXError(
        """eigenvector_centrality():
power iteration failed to converge in %d iterations."%(i+1))"""
    )


def large_clique_size(graph):
    """
    Function to return the largest clique size of the subgraph

    @itype  graph: nxGraph

    @rtype: int
    """
    return nx.algorithms.approximation.large_clique_size(graph)


def diameter(graph):
    """
    Function to return the lower bound on diameter of the subgraph

    @itype  graph: nxGraph

    @rtype: int
    """
    return nx.diameter(graph)


def communicability(graph):
    """
    Function to find communicability between all pairs of nodes in graph

    @itype  graph: nxGraph

    @rtype: dict of dict
    """
    return nx.communicability(graph)


def pagerank(graph):
    """
    Function to find pagerank for all nodes in the subgraph

    @itype  graph: nxGraph

    @rtype: dict
    """
    return nx.pagerank(graph)
