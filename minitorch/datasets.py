import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """
    Generates N random 2D points within the unit square [0,1] x [0,1].
    
    Args:
        N (int): Number of points to generate.
    
    Returns:
        List[Tuple[float, float]]: List of tuples representing (x1, x2) coordinates.
    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """
    Container for a dataset.

    Attributes:
        N (int): Number of points in the dataset.
        X (List[Tuple[float, float]]): List of input points (features).
        y (List[int]): Corresponding labels (0 or 1).
    """
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    Generates a simple linearly separable dataset.
    
    Args:
        N (int): Number of points to generate.
    
    Returns:
        Graph: Contains inputs X and binary labels y.
    
    Labels are 1 if x_1 < 0.5, otherwise 0. Useful for testing simple linear models.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Generates a dataset separable along the diagonal.
    
    Args:
        N (int): Number of points to generate.
    
    Returns:
        Graph: Contains inputs X and binary labels y.
    
    Labels are 1 if x_1 + x_2 < 0.5, otherwise 0. Tests linear models with diagonal separation.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    Generates a dataset split into two outer vertical bands.
    
    Args:
        N (int): Number of points to generate.
    
    Returns:
        Graph: Contains inputs X and binary labels y.
    
    Labels are 1 if x_1 < 0.2 or x_1 > 0.8, otherwise 0.
    Useful for testing models on non-central regions.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    Generates an XOR dataset.
    
    Args:
        N (int): Number of points to generate.
    
    Returns:
        Graph: Contains inputs X and binary labels y.
    
    Labels are 1 if x_1 and x_2 are on opposite sides of 0.5, otherwise 0.
    Non-linearly separable, useful for testing multi-layer networks.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    Generates a circular dataset.
    
    Args:
        N (int): Number of points to generate.
    
    Returns:
        Graph: Contains inputs X and binary labels y.
    
    Labels are 1 if the point lies outside a circle of radius sqrt(0.1) centered at (0.5, 0.5), otherwise 0.
    Useful for testing models on radial/non-linear decision boundaries.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Generates a two-spiral dataset.
    
    Args:
        N (int): Number of points to generate.
    
    Returns:
        Graph: Contains inputs X and binary labels y.
    
    Labels are 0 for the first spiral, 1 for the second.
    Useful for testing models on complex non-linear decision boundaries.
    """
    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
