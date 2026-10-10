import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mori_tanaka import core as mt


PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def isotropic_tensor(E, nu):
    """Return the full fourth-order isotropic stiffness tensor."""
    delta = np.eye(3)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return (
        lam * np.einsum("ij,kl->ijkl", delta, delta)
        + mu * np.einsum("ik,jl->ijkl", delta, delta)
        + mu * np.einsum("il,jk->ijkl", delta, delta)
    )


def isotropic_compliance_tensor(E, nu):
    """Return the full fourth-order isotropic compliance tensor."""
    delta = np.eye(3)
    return (
        (1 + nu)
        / (2 * E)
        * (
            np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        )
        - nu / E * np.einsum("ij,kl->ijkl", delta, delta)
    )


def eshelby_tensor(nu, aspect_ratio):
    """Return S_ijkl without a contracted-notation shear convention."""
    s = aspect_ratio
    I1 = (2 * s / (s**2 - 1) ** 1.5) * (
        s * np.sqrt(s**2 - 1) - np.arccosh(s)
    )
    Q = 3 / (8 * (1 - nu))
    R = (1 - 2 * nu) / (8 * (1 - nu))
    T = Q * (4 - 3 * I1) / (3 * (s**2 - 1))
    I3 = 4 - 2 * I1

    voigt = np.zeros((6, 6))
    voigt[0, 0] = Q + R * I1 + 0.75 * T
    voigt[1, 1] = voigt[0, 0]
    voigt[2, 2] = 4 / 3 * Q + R * I3 + 2 * s**2 * T
    voigt[0, 1] = Q / 3 - R * I1 + T / 4
    voigt[1, 0] = voigt[0, 1]
    voigt[0, 2] = -R * I1 - s**2 * T
    voigt[1, 2] = voigt[0, 2]
    voigt[2, 0] = -R * I3 - T
    voigt[2, 1] = voigt[2, 0]
    voigt[5, 5] = Q / 3 + R * I1 + T / 4
    voigt[3, 3] = 2 * R - I1 * R / 2 - (1 + s**2) * T / 2
    voigt[4, 4] = voigt[3, 3]

    tensor = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(PAIRS):
        for J, (k, l) in enumerate(PAIRS):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    tensor[ii, jj, kk, ll] = voigt[I, J]
    return tensor


def matrix_from_symmetric_components(values):
    matrix = np.zeros((3, 3))
    for value, (i, j) in zip(values, PAIRS):
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def symmetric_components(matrix):
    return np.array([matrix[i, j] for i, j in PAIRS])


def symmetric_operator_matrix(tensor):
    """Matrix of a fourth-order tensor acting on raw tensor components."""
    operator = np.zeros((6, 6))
    for column, (k, l) in enumerate(PAIRS):
        argument = np.zeros((3, 3))
        argument[k, l] = 1
        argument[l, k] = 1
        result = np.einsum("ijkl,kl->ij", tensor, argument)
        operator[:, column] = symmetric_components(result)
    return operator


def tensorial_unidirectional_stiffness(Ef, nuf, Em, num, vf, aspect_ratio):
    """Independent Mori-Tanaka calculation using full tensor contractions."""
    Cf = isotropic_tensor(Ef, nuf)
    Cm = isotropic_tensor(Em, num)
    delta_C = Cf - Cm
    matrix_compliance = isotropic_compliance_tensor(Em, num)
    S = eshelby_tensor(num, aspect_ratio)
    delta = np.eye(3)
    symmetric_identity = 0.5 * (
        np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
    )

    dilute_inverse = symmetric_identity + np.einsum(
        "ijkl,klmn,mnpq->ijpq", S, matrix_compliance, delta_C
    )
    dilute = np.linalg.inv(symmetric_operator_matrix(dilute_inverse))
    concentration = dilute @ np.linalg.inv((1 - vf) * np.eye(6) + vf * dilute)

    # Apply six engineering-strain load cases, but perform all constitutive
    # operations as full tensor contractions.
    stiffness = np.zeros((6, 6))
    for column in range(6):
        strain_components = np.zeros(6)
        strain_components[column] = 1 if column < 3 else 0.5
        strain = matrix_from_symmetric_components(strain_components)
        fiber_strain = matrix_from_symmetric_components(
            concentration @ strain_components
        )
        stress = np.einsum("ijkl,kl->ij", Cm, strain) + vf * np.einsum(
            "ijkl,kl->ij", delta_C, fiber_strain
        )
        stiffness[:, column] = symmetric_components(stress)
    return stiffness


def orientation_average_tensor(C, a2, a4):
    """Advani-Tucker orientation average in full index notation."""
    B1 = C[0, 0] + C[2, 2] - 2 * C[1, 2] - 4 * C[4, 4]
    B2 = C[1, 2] - C[0, 1]
    B3 = C[4, 4] - 0.5 * (C[0, 0] - C[0, 1])
    B4 = C[0, 1]
    B5 = 0.5 * (C[0, 0] - C[0, 1])
    delta = np.eye(3)
    return (
        B1 * a4
        + B2
        * (
            np.einsum("ij,kl->ijkl", a2, delta)
            + np.einsum("kl,ij->ijkl", a2, delta)
        )
        + B3
        * (
            np.einsum("ik,jl->ijkl", a2, delta)
            + np.einsum("il,jk->ijkl", a2, delta)
            + np.einsum("jk,il->ijkl", a2, delta)
            + np.einsum("jl,ik->ijkl", a2, delta)
        )
        + B4 * np.einsum("ij,kl->ijkl", delta, delta)
        + B5
        * (
            np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        )
    )


def engineering_stiffness_from_tensor(tensor):
    return np.array(
        [[tensor[i, j, k, l] for k, l in PAIRS] for i, j in PAIRS]
    )


def engineering_properties(C):
    compliance = np.linalg.inv(C)
    return np.array(
        [
            1 / compliance[0, 0],
            1 / compliance[1, 1],
            1 / compliance[2, 2],
            -compliance[0, 1] / compliance[0, 0],
            -compliance[0, 2] / compliance[0, 0],
            -compliance[1, 2] / compliance[1, 1],
            1 / compliance[5, 5],
            1 / compliance[4, 4],
            1 / compliance[3, 3],
        ]
    )


def tensorial_properties(C):
    compliance = np.linalg.inv(symmetric_operator_matrix(C))
    return np.array(
        [
            1 / compliance[0, 0],
            1 / compliance[1, 1],
            1 / compliance[2, 2],
            -compliance[1, 0] / compliance[0, 0],
            -compliance[2, 0] / compliance[0, 0],
            -compliance[2, 1] / compliance[1, 1],
            1 / (2 * compliance[5, 5]),
            1 / (2 * compliance[4, 4]),
            1 / (2 * compliance[3, 3]),
        ]
    )


class VoigtConsistencyTests(unittest.TestCase):
    def test_eshelby_transverse_component_matches_published_closed_form(self):
        """Check the axis-relabeled S_2233 expression in Pan et al. Eq. 35."""
        nu = 0.4
        for s in (1.01, 1.1, 2.0, 5.0, 10.0):
            u = s**2 - 1
            g = s / u**1.5 * (s * np.sqrt(u) - np.arccosh(s))
            published = (
                s**2 / (2 * u) - (1 - 2 * nu + 3 / (4 * u)) * g
            ) / (4 * (1 - nu))
            self.assertAlmostEqual(mt.eshelby(nu, s)[0, 1], published, places=12)

    def test_eshelby_exact_sphere_matches_published_solution(self):
        """Check the isotropic spherical-inclusion tensor in Pan et al. Eq. 32."""
        nu = 0.4
        normal_diagonal = (7 - 5 * nu) / (15 * (1 - nu))
        normal_cross = (5 * nu - 1) / (15 * (1 - nu))
        engineering_shear = 2 * (4 - 5 * nu) / (15 * (1 - nu))
        expected = np.zeros((6, 6))
        expected[:3, :3] = normal_cross
        np.fill_diagonal(expected[:3, :3], normal_diagonal)
        np.fill_diagonal(expected[3:, 3:], engineering_shear)

        np.testing.assert_allclose(mt.eshelby(nu, 1.0), expected, rtol=1e-14)
        np.testing.assert_allclose(
            mt.eshelby(nu, 1.0 + 1e-12), expected, rtol=1e-11
        )

    def test_eshelby_rejects_unsupported_oblate_aspect_ratio(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            mt.eshelby(0.4, 0.9)

    def test_eshelby_has_axis_three_transverse_isotropy(self):
        for s in (1.0, 1.01, 2.0, 20.0):
            S = mt.eshelby(0.4, s)
            self.assertAlmostEqual(S[0, 0], S[1, 1], places=13)
            self.assertAlmostEqual(S[0, 2], S[1, 2], places=13)
            self.assertAlmostEqual(S[2, 0], S[2, 1], places=13)
            self.assertAlmostEqual(S[3, 3], S[4, 4], places=13)
            # In engineering Voigt form the shear row is doubled, so the
            # tensor identity S_1212=(S_1111-S_1122)/2 becomes this relation.
            self.assertAlmostEqual(S[5, 5], S[0, 0] - S[0, 1], places=13)

    def test_isotropic_and_equivalent_orthotropic_inputs_match(self):
        E = 3.8e9
        nu = 0.4
        G = E / (2 * (1 + nu))
        isotropic = mt.isotropicC(E, nu)
        orthotropic = mt.orthotropicC(E, E, E, nu, nu, nu, G, G, G)
        np.testing.assert_allclose(isotropic, orthotropic, rtol=1e-14, atol=1e-5)

    def test_engineering_voigt_matches_full_tensor_calculation(self):
        Ef, nuf = 230e9, 0.2
        Em, num = 3.8e9, 0.4
        vf, aspect_ratio = 0.2, 20.0
        a2 = np.diag([1.0, 0.0, 0.0])

        Cf = mt.isotropicC(Ef, nuf)
        Cm = mt.isotropicC(Em, num)
        S = mt.eshelby(num, aspect_ratio)
        A = mt.strainConcentration(vf, S, Cf, Cm)
        voigt_unidirectional = Cm + vf * np.dot(Cf - Cm, A)
        voigt_average = mt.ornavg(
            voigt_unidirectional, mt.quadraticClosure(a2), a2
        )

        tensor_unidirectional = tensorial_unidirectional_stiffness(
            Ef, nuf, Em, num, vf, aspect_ratio
        )
        a4 = np.einsum("ij,kl->ijkl", a2, a2)
        tensor_average = orientation_average_tensor(
            tensor_unidirectional, a2, a4
        )

        np.testing.assert_allclose(
            voigt_unidirectional,
            tensor_unidirectional,
            rtol=1e-13,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            voigt_average,
            engineering_stiffness_from_tensor(tensor_average),
            rtol=1e-13,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            engineering_properties(voigt_average),
            tensorial_properties(tensor_average),
            rtol=1e-13,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
