import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mori_tanaka import core as mt


def homogenized_stiffness(
    Cf, Cm, vf, aspect_ratio, a2, matrix_poisson=0.4, closure="quadratic"
):
    S = mt.eshelby(matrix_poisson, aspect_ratio)
    A = mt.strainConcentration(vf, S, Cf, Cm)
    C_uni = Cm + vf * (Cf - Cm) @ A
    if closure == "linear":
        a4 = mt.linearClosure(a2, -1 / 35, 1 / 7)
    else:
        a4 = mt.quadraticClosure(a2)
    C_avg = mt.ornavg(C_uni, a4, a2)
    mt.validate_stiffness_matrix(C_avg, "homogenized stiffness")
    return C_avg


class InputValidationTests(unittest.TestCase):
    def test_isotropic_material_rejects_nonpositive_or_nonfinite_modulus(self):
        for E in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(E=E), self.assertRaises(ValueError):
                mt.isotropicC(E, 0.3)

    def test_isotropic_material_rejects_unstable_poisson_ratio(self):
        for nu in (-1.0, 0.5, 0.6, np.nan, np.inf):
            with self.subTest(nu=nu), self.assertRaises(ValueError):
                mt.isotropicC(3.8e9, nu)

    def test_orthotropic_material_rejects_nonpositive_modulus(self):
        with self.assertRaisesRegex(ValueError, "E2 must be positive"):
            mt.orthotropicC(10.0, 0.0, 5.0, 0.2, 0.2, 0.2, 2.0, 2.0, 2.0)

    def test_orthotropic_material_rejects_unstable_compliance(self):
        with self.assertRaisesRegex(ValueError, "positive definite"):
            mt.orthotropicC(1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0)

    def test_volume_fraction_rejects_values_outside_physical_interval(self):
        for vf in (-0.01, 1.01, np.nan, np.inf):
            with self.subTest(vf=vf), self.assertRaises(ValueError):
                mt.validate_volume_fraction(vf)

    def test_orientation_rejects_nonsymmetric_tensor(self):
        a2 = np.array([[0.5, 0.1, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "symmetric"):
            mt.validate_orientation_tensor(a2)

    def test_orientation_rejects_wrong_trace(self):
        with self.assertRaisesRegex(ValueError, "trace 1"):
            mt.validate_orientation_tensor(np.diag([0.4, 0.4, 0.1]))

    def test_orientation_rejects_negative_eigenvalue(self):
        a2 = np.array([[0.5, 1.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            mt.validate_orientation_tensor(a2)

    def test_orientation_rejects_nonfinite_component(self):
        a2 = np.diag([0.5, 0.5, 0.0])
        a2[0, 1] = a2[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            mt.validate_orientation_tensor(a2)

    def test_anisotropic_matrix_model_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires an isotropic matrix"):
            mt.validate_matrix_model(False)

    def test_transverse_isotropy_rejects_inconsistent_datasheet_values(self):
        with self.assertRaisesRegex(
            ValueError, r"supplied G23=5e\+09 Pa.*7.40741e\+09 Pa.*32.5%"
        ):
            mt.validate_transverse_isotropy(15e9, 0.0125, 5e9, "fiber")

    def test_transverse_isotropy_accepts_exact_and_rounded_values(self):
        required_G23 = 15e9 / (2 * (1 + 0.25))
        mt.validate_transverse_isotropy(15e9, 0.25, required_G23, "fiber")
        mt.validate_transverse_isotropy(
            15e9, 0.25, 1.005 * required_G23, "rounded fiber"
        )

    def test_isotropic_stiffness_obeys_shear_modulus_relation(self):
        E, nu = 3.8e9, 0.4
        C = mt.isotropicC(E, nu)
        G = C[5, 5]
        self.assertAlmostEqual(E, 2 * G * (1 + nu), places=5)

    def test_orthotropic_compliance_obeys_poisson_reciprocity(self):
        E1, E2, E3 = 230e9, 15e9, 12e9
        C = mt.orthotropicC(
            E1, E2, E3, 0.2, 0.25, 0.15, 9e9, 5e9, 7e9
        )
        compliance = np.linalg.inv(C)
        nu12 = -compliance[1, 0] / compliance[0, 0]
        nu21 = -compliance[0, 1] / compliance[1, 1]
        self.assertAlmostEqual(nu12 / E1, nu21 / E2, places=25)

    def test_invalid_stiffness_postconditions_are_rejected(self):
        nonsymmetric = np.eye(6)
        nonsymmetric[0, 1] = 0.1
        indefinite = np.eye(6)
        indefinite[0, 0] = -1.0
        nonfinite = np.eye(6)
        nonfinite[0, 0] = np.nan
        for matrix in (nonsymmetric, indefinite, nonfinite):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                mt.validate_stiffness_matrix(matrix)

    def test_valid_boundary_and_auxetic_isotropic_materials(self):
        for nu in (-0.2, 0.0, 0.49):
            C = mt.isotropicC(3.8e9, nu)
            mt.validate_stiffness_matrix(C)
        mt.validate_volume_fraction(0.0)
        mt.validate_volume_fraction(1.0)

    def test_valid_orthotropic_fiber(self):
        C = mt.orthotropicC(
            230e9, 15e9, 15e9, 0.2, 0.25, 0.2, 9e9, 6e9, 9e9
        )
        mt.validate_transverse_isotropy(15e9, 0.25, 6e9, "fiber")
        mt.validate_stiffness_matrix(C)

    def test_valid_orientation_states(self):
        states = (
            np.eye(3) / 3,
            np.diag([1.0, 0.0, 0.0]),
            np.array([[0.5, -0.2, 0.0], [-0.2, 0.5, 0.0], [0.0, 0.0, 0.0]]),
        )
        for a2 in states:
            with self.subTest(a2=a2):
                mt.validate_orientation_tensor(a2)
                mt.quadraticClosure(a2)

    def test_valid_default_composite_case(self):
        Cf = mt.isotropicC(230e9, 0.2, "fiber")
        Cm = mt.isotropicC(3.8e9, 0.4, "matrix")
        C = homogenized_stiffness(
            Cf, Cm, 0.2, 20.0, np.diag([1.0, 0.0, 0.0])
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(C) > 0.0))

    def test_valid_spherical_random_orientation_case(self):
        Cf = mt.isotropicC(70e9, 0.22, "particle")
        Cm = mt.isotropicC(3.0e9, 0.35, "matrix")
        C = homogenized_stiffness(
            Cf, Cm, 0.15, 1.0, np.eye(3) / 3,
            matrix_poisson=0.35, closure="linear"
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(C) > 0.0))

    def test_valid_orthotropic_fiber_composite_case(self):
        Cf = mt.orthotropicC(
            230e9, 15e9, 15e9, 0.2, 0.25, 0.2, 9e9, 6e9, 9e9,
            "fiber",
        )
        mt.validate_transverse_isotropy(15e9, 0.25, 6e9, "fiber")
        Cm = mt.isotropicC(3.8e9, 0.4, "matrix")
        a2 = np.array(
            [[0.5, -0.2, 0.0], [-0.2, 0.5, 0.0], [0.0, 0.0, 0.0]]
        )
        C = homogenized_stiffness(Cf, Cm, 0.3, 5.0, a2)
        self.assertTrue(np.all(np.linalg.eigvalsh(C) > 0.0))


if __name__ == "__main__":
    unittest.main()
