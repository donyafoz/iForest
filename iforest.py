
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import multiprocessing as mp

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []

        self.c = self.tree_avg_length(self.sample_size)

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        height_limit = int(np.ceil(np.log2(self.sample_size)))

        for n in range(self.n_trees):
            # sample rows from X
            sample_rows = np.random.choice(X.shape[0],
                                            size=self.sample_size,
                                            replace=False)
            tree = IsolationTree(height_limit)
            tree.fit(X[sample_rows], improved)

            self.trees.append(tree)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        paths = np.zeros(X.shape[0], dtype=float)
        with mp.Pool(8) as p:
            paths = p.map(self._path_len, X)

        return np.array(paths)

    def _path_len(self, x):
        length = 0.0
        for tree in self.trees:
            length += self.tree_path_length(x, tree.root, 0)
        mean = length / self.n_trees
        return mean

    def tree_avg_length(self, size):
        if size > 2:
            harmonic_number = np.log(size - 1.) + 0.5772156649
            return (2. * harmonic_number) - (2. * (size - 1.) / size)

        elif size == 2:
            return 1.

        else:
            return 0.


    def tree_path_length(self, X, root, current_path_length):
        if isinstance(root, exNode):
            return current_path_length + self.tree_avg_length(root.size)

        split_at = root.q

        if X[split_at] < root.p:
            return self.tree_path_length(X, root.left, current_path_length + 1)

        else:
            return self.tree_path_length(X, root.right, current_path_length + 1)


    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        path_lengths = self.path_length(X)
        return 2.0 ** (-path_lengths / self.c)

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        s = scores.copy()
        s[s < threshold] = 0
        s[s >= threshold] = 1

        return s

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.n_nodes = 0

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = self.make_node(X, 0, improved)

        return self.root

    def make_node(self, X, current_height, improved):
        self.n_nodes += 1
        if current_height >= self.height_limit or X.shape[0] <= 1:
            return exNode(X.shape[0])

        if improved:
            qs = np.random.randint(0, X.shape[1], 2)

            best_q = qs[0]
            best_p = 0
            best__ = 0

            leftX = None
            rightX = None
            for q in qs:
                minimum = X[:,q].min()
                maximum = X[:,q].max()
                p = np.random.uniform(minimum, maximum)

                tmp_leftX = X[X[:,q]<p]
                tmp_rightX = X[X[:,q]>=p]

                left_len = len(tmp_leftX)
                right_len = len(tmp_rightX)

                var = abs(left_len - right_len)

                if var >= best__:
                    best__ = var
                    best_q = q
                    best_p = p

                    leftX = tmp_leftX
                    rightX = tmp_rightX

            q = best_q
            p = best_p

        else:
            q = np.random.randint(0, X.shape[1])
            minimum = X[:,q].min()
            maximum = X[:,q].max()
            p = np.random.uniform(minimum, maximum)

            leftX = X[X[:,q]<p]
            rightX = X[X[:,q]>=p]

        left = self.make_node(leftX, current_height+1, improved)
        right = self.make_node(rightX, current_height+1, improved)

        return inNode(left, right, q, p)


class inNode:
    def __init__(self, left, right, q, p):
        self.left = left
        self.right = right
        self.q = q
        self.p = p

class exNode:
    def __init__(self, size):
        self.size = size


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    TPR = 0.0
    FPR = 0.0
    y2 = y.reset_index()

    while TPR < desired_TPR:
        temp = scores.copy()
        temp[temp < threshold] = 0
        temp[temp >= threshold] = 1


        confusion = confusion_matrix(y, temp)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        threshold -= 0.01

    return threshold+0.01, FPR

