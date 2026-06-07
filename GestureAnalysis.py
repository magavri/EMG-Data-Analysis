"""
EMG Gesture Classification and Feature Space Analysis
=====================================================

This script performs:
- MAV feature extraction
- Gesture-set optimization
- LDA classification
- Force variability analysis
- Confusion matrix visualization
- MDS feature-space projection

"""

from __future__ import annotations

import itertools
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score, confusion_matrix

from statannotations.Annotator import Annotator


# =============================================================================
# Configuration
# =============================================================================

FS = 2048
WINDOW_LENGTH = int(0.2 * FS)
OVERLAP = int(0.1 * FS)
STEP = WINDOW_LENGTH - OVERLAP

LOWPASS_CUTOFF = 5
FILTER_ORDER = 4

RANDOM_SEED = 42

DATA_DIR = Path("Data\\Pre-processed\\")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def lowpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: int,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter.
    """

    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, signal)


def compute_bhattacharyya_distance(
    class1: np.ndarray,
    class2: np.ndarray,
    regularization: float = 1e-8
) -> float:
    """
    Compute Bhattacharyya distance between two feature sets.
    """

    mean_diff = np.mean(class1, axis=0) - np.mean(class2, axis=0)

    cov1 = np.cov(class1, rowvar=False)
    cov2 = np.cov(class2, rowvar=False)

    cov_avg = (cov1 + cov2) / 2

    cov_inv = np.linalg.inv(cov_avg)

    mahal_term = 0.125 * mean_diff @ cov_inv @ mean_diff

    det_term = 0.5 * np.log(
        np.linalg.det(cov_avg) /
        np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return float(mahal_term + det_term)


def compute_set_distances(
    features: Dict[str, np.ndarray],
    gesture_set: List[str]
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise gesture distances.
    """

    gesture_pairs = list(itertools.combinations(gesture_set, 2))

    distances = {}

    for pair in gesture_pairs:
        distances[pair] = compute_bhattacharyya_distance(
            features[pair[0]],
            features[pair[1]]
        )

    return distances


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_mav_features(
    data: np.ndarray
) -> Tuple[dict, dict]:
    """
    Extract MAV features from segmented EMG data.
    """

    gesture_ids = data.dtype.names

    features = {
        gesture: {trial: [] for trial in range(3)}
        for gesture in gesture_ids
    }

    combined_features = {
        gesture: []
        for gesture in gesture_ids
    }

    for gesture in gesture_ids:

        trial_features = []

        for trial in range(3):

            signal = data[gesture][0][trial]

            windows = np.squeeze(
                sliding_window_view(
                    signal,
                    window_shape=(WINDOW_LENGTH, signal.shape[1])
                )[::STEP]
            )

            mav = np.mean(np.abs(windows), axis=1)

            features[gesture][trial] = mav
            trial_features.append(mav)

        combined_features[gesture] = np.vstack(trial_features)

    return features, combined_features


# =============================================================================
# Classification
# =============================================================================

def run_lda_classification(
    gesture_set: List[str],
    features: dict
) -> Tuple[float, list, list]:
    """
    Run leave-one-trial-out LDA classification.
    """

    trial_order = np.array([
        [0, 1, 2],
        [1, 2, 0],
        [0, 2, 1]
    ])

    accuracies = []

    full_true = []
    full_pred = []

    for trial in trial_order:

        training = []
        testing = []

        y_train = []
        y_test = []

        for gesture in gesture_set:

            train_data = np.vstack([
                features[gesture][trial[0]],
                features[gesture][trial[1]]
            ])

            test_data = features[gesture][trial[2]]

            training.append(train_data)
            testing.append(test_data)

            y_train.append(
                np.repeat(gesture, train_data.shape[0])
            )

            y_test.append(
                np.repeat(gesture, test_data.shape[0])
            )

        X_train = np.vstack(training)
        X_test = np.vstack(testing)

        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        n_classes = len(gesture_set)

        lda = LinearDiscriminantAnalysis(
            priors=[1 / n_classes] * n_classes
        )

        lda.fit(X_train, y_train)

        predictions = lda.predict(X_test)

        accuracies.append(
            accuracy_score(y_test, predictions)
        )

        full_true.append(y_test)
        full_pred.append(predictions)

    return np.mean(accuracies) * 100, full_true, full_pred


# =============================================================================
# Gesture Selection
# =============================================================================

def get_pair_distance(BD, a, b):
    """Return symmetric distance safely."""
    return BD.get((a, b)) or BD.get((b, a))


def compute_brute_force_set(
    features: dict,
    n_classes: List[int],
    gesture_ids: List[str],
    n_jobs: int = -1
) -> dict:
    """
    Find optimal gesture combinations using brute-force search.
    """

    distances = compute_set_distances(features, gesture_ids)

    optimal_sets = {}

    for n in n_classes:

        combinations_list = list(
            itertools.combinations(gesture_ids, n)
        )

        def min_pair_distance(combo):

            pairs = list(itertools.combinations(combo, 2))

            pair_distances = [
                get_pair_distance(distances, a, b)
                for a, b in pairs
            ]

            return np.min(pair_distances)
        scores = Parallel(n_jobs=n_jobs)(
            delayed(min_pair_distance)(combo)
            for combo in combinations_list
        )

        best_idx = np.argmax(scores)

        optimal_sets[n] = combinations_list[best_idx]

    return optimal_sets


# =============================================================================
# Force Variability
# =============================================================================

def compute_force_variability(
    data: np.ndarray
) -> np.ndarray:
    """
    Compute coefficient of variation for each gesture.
    """

    gesture_ids = data.dtype.names

    cv = np.zeros((len(gesture_ids), 3))

    for g, gesture in enumerate(gesture_ids):

        for trial in range(3):

            rectified = np.mean(
                np.abs(data[gesture][0][trial]),
                axis=1
            )

            envelope = lowpass_filter(
                rectified,
                cutoff=LOWPASS_CUTOFF,
                fs=FS,
                order=FILTER_ORDER
            )

            cv[g, trial] = (
                np.std(envelope) /
                np.mean(envelope)
            )

    return np.mean(cv, axis=1) * 100


# =============================================================================
# Visualization
# =============================================================================

def plot_force_variability(df: pd.DataFrame):

    fig, ax = plt.subplots(figsize=(5, 5))

    palette = ["#4C72B0", "#bda5b6"]

    sns.boxplot(
        x="group",
        y="value",
        data=df,
        palette=palette,
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        x="group",
        y="value",
        data=df,
        hue="subject",
        dodge=False,
        size=7,
        ax=ax
    )

    stat, p = mannwhitneyu(
        df[df["group"] == "Limb loss"]["value"],
        df[df["group"] == "Able-bodied"]["value"]
    )

    annotator = Annotator(
        ax,
        [("Limb loss", "Able-bodied")],
        data=df,
        x="group",
        y="value"
    )

    annotator.set_pvalues([p])
    annotator.annotate()

    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_xlabel(None)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    gesture_groups: dict,
    features: dict
):

    fig, axes = plt.subplots(
        len(gesture_groups),
        1,
        figsize=(5, 25)
    )

    for i, group in enumerate(gesture_groups.values()):

        _, true_labels, predictions = run_lda_classification(
            group,
            features
        )

        cm = confusion_matrix(
            np.hstack(true_labels),
            np.hstack(predictions)
        )

        labels = np.unique(true_labels[0])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[i]
        )

    plt.tight_layout()
    plt.show()


def plot_mds_projection(
    features: dict,
    gesture_ids: list
):

    distances = compute_set_distances(
        features,
        gesture_ids
    )

    graph = nx.Graph()

    for (u, v), d in distances.items():
        graph.add_edge(u, v, weight=d)

    nodes = list(graph.nodes())

    n_nodes = len(nodes)

    node_to_idx = {
        node: i for i, node in enumerate(nodes)
    }

    D = np.zeros((n_nodes, n_nodes))

    shortest_paths = dict(
        nx.all_pairs_dijkstra_path_length(
            graph,
            weight="weight"
        )
    )

    for u in nodes:
        for v in nodes:
            D[
                node_to_idx[u],
                node_to_idx[v]
            ] = shortest_paths[u][v]

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=RANDOM_SEED,
        n_init=1
    )

    coords = mds.fit_transform(D)

    plt.figure(figsize=(6, 5))

    plt.scatter(coords[:, 0], coords[:, 1])

    for i, node in enumerate(nodes):

        plt.text(
            coords[i, 0],
            coords[i, 1],
            node
        )

    plt.title("Feature Space Projection")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():

    subject = "S2"

    file_path = DATA_DIR / f"{subject}.mat"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}"
        )

    logger.info(f"Loading {subject}")

    temp = loadmat(file_path)

    if "segmentedDataG" not in temp:
        raise KeyError(
            "Expected variable 'segmentedDataG' not found."
        )

    data = temp["segmentedDataG"]

    gesture_ids = list(data.dtype.names)

    logger.info("Extracting features")

    features, combined_features = extract_mav_features(data)

    logger.info("Running gesture optimization")

    n_classes = [4, 6, 8, 10]

    brute_sets = compute_brute_force_set(
        combined_features,
        n_classes,
        gesture_ids
    )

    logger.info("Running classification")

    for n in n_classes:

        accuracy, _, _ = run_lda_classification(
            brute_sets[n],
            features
        )

        logger.info(
            f"{n} classes: "
            f"{accuracy:.2f}%"
        )

    logger.info("Generating MDS visualization")

    plot_mds_projection(
        combined_features,
        gesture_ids
    )

    logger.info("Done")


if __name__ == "__main__":
    main()
