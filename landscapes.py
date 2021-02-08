from abc import ABC, abstractmethod
import math
from typing import Sequence, Type

import numpy as np


class Landscape(ABC):
    def __init__(self, name: str, dim: int, domain: np.ndarray):
        self.name = name
        self.dim = dim
        self.domain = domain

    @abstractmethod
    def __call__(self, x: Sequence[float]) -> float:
        pass

    def sample(self, N: int) -> np.ndarray:
        """sample N points uniformly from this objective's domain"""
        return np.random.uniform(
            low=self.domain[:,0], high=self.domain[:,1], size=(N, self.dim)
        )


class Ackley(Landscape):
    def __init__(self, dim: int = 2,
                 a: float = 20., b: float = 0.2, c: float = 2*math.pi,
                 *args, **kwargs):
        self.a = a
        self.b = b
        self.c = c

        domain = np.array([(-32.768, 32.768) for _ in range(dim)])

        super().__init__(name='ackley', dim=dim, domain=domain)

    def __call__(self, X: np.ndarray) -> float:
        if len(X.shape) == 1:
            X = np.reshape(X, (1, len(X)))
        X = X[:,:self.dim]

        U = -self.b * np.sqrt(1/self.dim * np.sum(X*X, axis=1))
        V = 1/self.dim * np.sum(np.cos(self.c * X), axis=1)

        return -self.a*np.exp(U) - np.exp(V) + self.a + np.exp(1)


class Bukin6(Landscape):
    def __init__(self, *args, **kwargs):
        dim = 2
        domain = np.array([(-15, -5), (-3, 3)])

        super().__init__(name='bukin6', dim=dim, domain=domain)

    def __call__(self, X) -> float:
        if len(X.shape) == 1:
            X = np.reshape(X, (1, len(X)))

        U = 100 * np.sqrt(np.abs(X[:,1] - 0.01 * X[:,0]**2))
        V = 0.01 * np.abs(X[:,0] + 10)
        
        return U + V


class Langermann(Landscape):
    def __init__(self, dim: int = 2, m: int = 5,
                 c: np.ndarray = np.array([1,2,5,2,3]),
                 A: np.ndarray = np.array([[3,5],[5,2],[2,1],[1,4],[7,9]]),
                 *args, **kwargs):
        self.dim = dim
        self.m = m

        self.c = c
        if self.c.shape[0] < self.m or len(self.c.shape) > 1:
            raise ValueError('c must be 1-D array of length m!')

        self.A = A
        if self.A.shape[0] < self.m or self.A.shape[1] != self.dim:
            raise ValueError('A must be 2-D array of shape m x D')

        self.domain = np.array([(0, 10) for _ in range(dim)])

    def __call__(self, X) -> float:
        if len(X.shape) == 1:
            X = np.reshape(X, (1, len(X)))
        X = X[:,:self.dim]

        a = -1 / np.pi

        Y = np.empty((X.shape[0], self.m))
        for j in range(self.m):
            U = X - self.A[j]
            V = np.sum(U*U, axis=1)

            Y[:, j] = self.c[j] * np.exp(a * V) * np.cos(np.pi * V)

        return np.sum(Y, axis=1)


class Rastrigin(Landscape):
    def __init__(self, dim: int = 2, *args, **kwargs):
        dim = dim
        domain = np.array([(-5.12, 5.12) for _ in range(dim)])

        super().__init__(name='bukin6', dim=dim, domain=domain)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = np.reshape(X, (1, len(X)))
        X = X[:,:self.dim]

        return 10*self.dim + np.sum(X*X - 10*np.cos(2*np.pi*X), axis=1)


class Schaffer2(Landscape):
    def __init__(self, *args, **kwargs):
        dim = 2
        domain = np.array([(-100, 100) for _ in range(dim)])

        super().__init__(name='schaffer2', dim=dim, domain=domain)

    def __call__(self, X) -> float:
        if len(X.shape) == 1:
            X = np.reshape(X, (1, len(X)))
        
        X_0_sq = X[:, 0]**2
        X_1_sq = X[:, 1]**2

        U = np.sin(X_0_sq - X_1_sq)**2 - 0.5
        V = (1 + 0.001*(X_0_sq + X_1_sq))**2

        return 0.5 + U / V


def build_landscape(landscape: str = 'ackley',
                    dimension: int = 2) -> Type[Landscape]:
    return {
        'ackley': Ackley,
        'bukin': Bukin6,
        'langermann': Langermann,
        'rastrigin': Rastrigin,
        'schaffer': Schaffer2
    }[landscape](dim=dimension)