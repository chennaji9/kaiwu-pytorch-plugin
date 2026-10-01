# -*- coding: utf-8 -*-
"""Global-RNG isolation tests for the Kaiwu SA backend in maifs/qubo.py."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import kaiwu  # pylint: disable=wrong-import-position
from kaiwu.torch_plugin.maifs.qubo import (  # pylint: disable=wrong-import-position
    _solve_ising_sa,
)


class TestSaGlobalRngIsolation(unittest.TestCase):
    """Seeding Kaiwu SA must not change the caller's global RNG stream."""

    @staticmethod
    def _fake_optimizer_class(draws):
        class FakeOptimizer:
            """Records draws from the seeded global stream, like Kaiwu SA."""

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def solve(self, matrix):
                del matrix
                draws.append(np.random.random(3))
                return np.array([[1, -1, 1, 1]])

        return FakeOptimizer

    def test_solver_does_not_leak_seeded_rng_state(self):
        matrix = np.zeros((4, 4))
        draws = []
        np.random.seed(123)
        state_before = np.random.get_state()
        expected_next = np.random.random(5)
        np.random.set_state(state_before)

        with mock.patch.object(
            kaiwu.classical,
            "SimulatedAnnealingOptimizer",
            self._fake_optimizer_class(draws),
        ):
            _solve_ising_sa(matrix, initial_binary=np.array([0, 1, 1]), random_state=7)

        # The caller's stream is exactly what it would have been without the call.
        np.testing.assert_array_equal(np.random.random(5), expected_next)
        self.assertEqual(len(draws), 1)

    def test_solver_keeps_its_own_reproducibility(self):
        matrix = np.zeros((4, 4))
        draws_a, draws_b = [], []
        for draws in (draws_a, draws_b):
            with mock.patch.object(
                kaiwu.classical,
                "SimulatedAnnealingOptimizer",
                self._fake_optimizer_class(draws),
            ):
                _solve_ising_sa(matrix, initial_binary=np.array([0, 1, 1]), random_state=7)
        # Same random_state still yields identical seeded draws inside the solver.
        np.testing.assert_array_equal(draws_a[0], draws_b[0])

    def test_rng_state_restored_when_solver_raises(self):
        matrix = np.zeros((4, 4))

        class FailingOptimizer:
            """Simulates a Kaiwu SA failure after consuming the seeded stream."""

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def solve(self, matrix):
                del matrix
                np.random.random(3)
                raise RuntimeError("solver failed")

        np.random.seed(123)
        state_before = np.random.get_state()
        expected_next = np.random.random(5)
        np.random.set_state(state_before)

        with mock.patch.object(
            kaiwu.classical, "SimulatedAnnealingOptimizer", FailingOptimizer
        ):
            with self.assertRaises(RuntimeError):
                _solve_ising_sa(matrix, initial_binary=np.array([0, 1, 1]), random_state=7)

        np.testing.assert_array_equal(np.random.random(5), expected_next)


if __name__ == "__main__":
    unittest.main()
