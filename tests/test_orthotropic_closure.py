import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mori_tanaka import core as mt


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.dot(cross, cross)
    )


def transformed_closure(A4, q):
    R = mt.R_sigma(q)
    return np.dot(R, np.dot(A4, R.T))


class OrthotropicClosureTests(unittest.TestCase):
    def assert_objective(self, a2, q, atol=2e-14):
        A4 = mt.orthoClosure(a2)
        rotated_a2 = np.dot(q, np.dot(a2, q.T))
        rotated_A4 = mt.orthoClosure(rotated_a2)
        np.testing.assert_allclose(
            rotated_A4, transformed_closure(A4, q), rtol=0.0, atol=atol
        )

    def test_coordinate_permutations_are_objective(self):
        a2 = np.diag([0.7, 0.2, 0.1])
        permutations = (
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        )
        for q in permutations:
            with self.subTest(q=q):
                self.assert_objective(a2, q)

    def test_general_rotation_is_objective(self):
        self.assert_objective(
            np.diag([0.7, 0.2, 0.1]), rotation([1.0, 2.0, 3.0], 0.73)
        )

    def test_nearly_degenerate_state_is_objective(self):
        q = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assert_objective(np.diag([0.500001, 0.499999, 0.0]), q)

    def test_isotropic_state_has_exact_isotropic_fourth_moment(self):
        expected = np.zeros((6, 6))
        expected[:3, :3] = 1 / 15
        np.fill_diagonal(expected[:3, :3], 1 / 5)
        np.fill_diagonal(expected[3:, 3:], 1 / 15)
        np.testing.assert_allclose(
            mt.orthoClosure(np.eye(3) / 3), expected, rtol=0.0, atol=1e-15
        )

    def test_axisymmetric_states_do_not_depend_on_degenerate_basis(self):
        self.assert_objective(
            np.diag([0.5, 0.5, 0.0]), rotation([0.0, 0.0, 1.0], 0.37)
        )
        self.assert_objective(
            np.diag([0.8, 0.1, 0.1]), rotation([1.0, 0.0, 0.0], 0.61)
        )

    def test_axisymmetric_states_are_objective_under_general_rotation(self):
        q = rotation([1.0, -2.0, 0.5], 0.44)
        self.assert_objective(np.diag([0.5, 0.5, 0.0]), q)
        self.assert_objective(np.diag([0.8, 0.1, 0.1]), q)

    def test_closure_contracts_to_second_order_orientation(self):
        q = rotation([2.0, -1.0, 0.5], 0.42)
        states = (
            np.diag([0.7, 0.2, 0.1]),
            np.eye(3) / 3,
            np.diag([0.5, 0.5, 0.0]),
            np.dot(q, np.dot(np.diag([0.8, 0.1, 0.1]), q.T)),
        )
        for a2 in states:
            with self.subTest(a2=a2):
                A4 = mt.orthoClosure(a2)
                contracted = np.sum(A4[:, :3], axis=1)
                expected = np.array(
                    [a2[0, 0], a2[1, 1], a2[2, 2], a2[1, 2], a2[0, 2], a2[0, 1]]
                )
                np.testing.assert_allclose(
                    contracted, expected, rtol=0.0, atol=2e-14
                )

    def test_downstream_stiffness_is_permutation_invariant(self):
        a2 = np.diag([0.7, 0.2, 0.1])
        q = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated_a2 = np.dot(q, np.dot(a2, q.T))

        Cf = mt.isotropicC(230e9, 0.2, "fiber")
        Cm = mt.isotropicC(3.8e9, 0.4, "matrix")
        S = mt.eshelby(0.4, 20.0)
        A = mt.strainConcentration(0.2, S, Cf, Cm)
        C_uni = Cm + 0.2 * np.dot(Cf - Cm, A)

        C_avg = mt.ornavg(C_uni, mt.orthoClosure(a2), a2)
        rotated_C_avg = mt.ornavg(
            C_uni, mt.orthoClosure(rotated_a2), rotated_a2
        )
        np.testing.assert_allclose(
            rotated_C_avg,
            transformed_closure(C_avg, q),
            rtol=1e-14,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
