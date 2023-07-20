# pylint: disable=no-member

"""
module to get tda features for classifiers
"""
import numpy as np
from gtda.homology import VietorisRipsPersistence
from sklearn.cluster import DBSCAN
import gudhi
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from statistics import variance, mean
from scipy.stats import entropy


def convert_graph_to_adjacency_matrix(graph_data):
    nodes = graph_data["nodes"]
    links = graph_data["links"]

    node_id_to_index = {node["id"]: index for index, node in enumerate(nodes)}

    num_nodes = len(nodes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for link in links:
        source_index = node_id_to_index[link["source"]]
        target_index = node_id_to_index[link["target"]]
        weight = link["weight"]
        adjacency_matrix[source_index, target_index] = weight

    return adjacency_matrix


def calculate_persistence_diagram(adjacency_matrix):
    try:
        # Compute Vietoris-Rips persistence
        vrp = VietorisRipsPersistence(
            metric="precomputed", homology_dimensions=[0, 1, 2], n_jobs=-1
        )
        vrp.fit([adjacency_matrix])
        persistence_diagrams = vrp.transform([adjacency_matrix])

        rips_complex = gudhi.RipsComplex(points=persistence_diagrams[0])
        simplex_tree = rips_complex.create_simplex_tree()
        persistence_bars = simplex_tree.persistence()
        number_of_homology_groups = len(persistence_bars)

        bar_lengths = []
        for _, persist_bar in persistence_bars:
            current_max = persist_bar[1] - persist_bar[0]
            if current_max == np.inf:
                current_max = 10
            bar_lengths.append(current_max)

        # Compute additional TDA features
        num_points = persistence_diagrams[0].shape[0]
        mean_lifetime = np.mean(
            persistence_diagrams[0][:, 1] - persistence_diagrams[0][:, 0]
        )
        max_lifetime = np.max(
            persistence_diagrams[0][:, 1] - persistence_diagrams[0][:, 0]
        )
        num_clusters = len(np.unique(DBSCAN().fit_predict(persistence_diagrams[0])))

        # Perform dimensionality reduction using PCA
        pca = make_pipeline(StandardScaler(), PCA(n_components=2))
        persistence_features = pca.fit_transform(persistence_diagrams[0])

        tda_features = [
            persistence_features.flatten().mean(),
            num_points,
            mean_lifetime,
            max_lifetime,
            num_clusters,
            number_of_homology_groups,
            sum(bar_lengths),
            mean(bar_lengths),
            variance(bar_lengths),
            entropy(bar_lengths, 2),
        ]

        return pd.Series(tda_features)

    except:
        return pd.Series([None, None, None, None, None, None, None, None, None, None])
