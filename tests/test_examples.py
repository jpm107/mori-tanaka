"""End-to-end examples checked against independent analytical references."""

import pytest

from mori_tanaka import effective_properties


def test_random_spherical_glass_polyester_against_closed_form_mori_tanaka():
    # Constituent data: Pan, "A re-formulation of the Mori-Tanaka method for
    # predicting material properties of fiber-reinforced polymers/composites"
    # (2019), Table 3. Reference values are a separate hand calculation using
    # Benveniste (1987), Eqs. (31)-(32), for spherical isotropic inclusions:
    # K=7.053763375176894 GPa and G=0.9223891229167346 GPa, followed by
    # E=9KG/(3K+G) and nu=(3K-2G)/(2(3K+G)).
    out = effective_properties(
        fiber_e1=70.3,
        fiber_nu12=0.21,
        matrix_e=1.72,
        matrix_nu=0.45,
        fiber_volume_fraction=0.2,
        aspect_ratio=1.0,
        orientation_a11=1.0 / 3.0,
        orientation_a22=1.0 / 3.0,
        closure="linear",
    )

    assert out["E1"] == pytest.approx(2.6515885716888055, rel=1e-11)
    assert out["E2"] == pytest.approx(2.6515885716888055, rel=1e-11)
    assert out["E3"] == pytest.approx(2.6515885716888055, rel=1e-11)
    assert out["nu12"] == pytest.approx(0.4373481352989505, rel=1e-11)
    assert out["nu13"] == pytest.approx(0.4373481352989505, rel=1e-11)
    assert out["nu23"] == pytest.approx(0.4373481352989505, rel=1e-11)
    assert out["G12"] == pytest.approx(0.9223891229167346, rel=1e-11)
    assert out["G13"] == pytest.approx(0.9223891229167346, rel=1e-11)
    assert out["G23"] == pytest.approx(0.9223891229167346, rel=1e-11)


def test_aligned_short_glass_epoxy_against_full_tensor_calculation():
    # Constituent data: Pan (2019), Table 1. Fixed reference values were
    # calculated independently in full fourth-order tensor notation from
    # Pan Eq. (4) and the published prolate Eshelby components, then rotated
    # from the local inclusion axis 3 to a perfectly aligned global axis 1.
    out = effective_properties(
        fiber_e1=72.4,
        fiber_nu12=0.20,
        matrix_e=2.76,
        matrix_nu=0.35,
        fiber_volume_fraction=0.2,
        aspect_ratio=20.0,
        orientation_a11=1.0,
        orientation_a22=0.0,
        closure="quadratic",
    )

    expected = {
        "E1": 13.345847129579713,
        "E2": 4.134714561269642,
        "E3": 4.134714561269642,
        "nu12": 0.3233059564522415,
        "nu13": 0.3233059564522416,
        "nu23": 0.4650901289771514,
        "G12": 1.4965949172258228,
        "G13": 1.4965949172258228,
        "G23": 1.4110785676224167,
    }
    assert out == pytest.approx(expected, rel=1e-10)


def test_oriented_transversely_isotropic_fibers_against_tensor_reference():
    # Independent full-tensor reference: axis-3 prolate Eshelby tensor and
    # Mori-Tanaka concentration, with the ORW coefficients from Chung and
    # Kwon, "Improved model of orthotropic closure approximation for flow
    # induced fiber orientation" (2001), Appendix. The calculation uses
    # a = diag(0.7, 0.2, 0.1) and enforces nu_ta/Et = nu_at/Ea.
    out = effective_properties(
        fiber_e1=230.0,
        fiber_e2=15.0,
        fiber_nu12=0.2,
        fiber_nu23=0.25,
        fiber_g12=9.0,
        fiber_g23=6.0,
        fiber_isotropic=False,
        matrix_e=3.8,
        matrix_nu=0.4,
        fiber_volume_fraction=0.2,
        aspect_ratio=20.0,
        orientation_a11=0.7,
        orientation_a22=0.2,
        orientation_a12=0.0,
        orientation_a13=0.0,
        orientation_a23=0.0,
        closure="orthotropic",
    )

    expected = {
        "E1": 17.941344528110008,
        "E2": 7.440634601919103,
        "E3": 6.517906703320624,
        "nu12": 0.42028115126824334,
        "nu13": 0.38996174188417454,
        "nu23": 0.48982777300432295,
        "G12": 3.5829684868232676,
        "G13": 2.7525979520842143,
        "G23": 2.2413360746347886,
    }
    assert out == pytest.approx(expected, rel=1e-9)
