"""Public, user-facing calculations for the Mori-Tanaka package."""

from . import core


def _transversely_isotropic_fiber_stiffness(
    fiber_e1: float,
    fiber_e2: float | None,
    fiber_nu12: float,
    fiber_nu23: float | None,
    fiber_g12: float | None,
    fiber_g23: float | None,
):
    values = {
        "fiber_e2": fiber_e2,
        "fiber_nu23": fiber_nu23,
        "fiber_g12": fiber_g12,
        "fiber_g23": fiber_g23,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "transversely isotropic fiber properties require "
            + ", ".join(missing)
        )

    core.validate_transverse_isotropy(
        fiber_e2, fiber_nu23, fiber_g23, "fiber"
    )
    # The Eshelby tensor and orientation-averaging coefficients use local
    # axis 3 as the inclusion length direction. Public fiber notation follows
    # the usual datasheet convention in which E1 and nu12 are axial. Relabel
    # the transversely isotropic constants into the local axis-3 basis and use
    # Poisson reciprocity for transverse-to-axial loading.
    transverse_to_axial_nu = fiber_nu12 * fiber_e2 / fiber_e1
    return core.orthotropicC(
        fiber_e2,
        fiber_e2,
        fiber_e1,
        fiber_nu23,
        transverse_to_axial_nu,
        transverse_to_axial_nu,
        fiber_g23,
        fiber_g12,
        fiber_g12,
        "fiber",
    )


def effective_properties(
    fiber_e1: float,
    fiber_nu12: float,
    matrix_e: float,
    matrix_nu: float,
    fiber_volume_fraction: float,
    aspect_ratio: float,
    orientation_a11: float,
    orientation_a22: float,
    *,
    orientation_a12: float = 0.0,
    orientation_a13: float = 0.0,
    orientation_a23: float = 0.0,
    closure: str = "linear",
    fiber_isotropic: bool = True,
    fiber_e2: float | None = None,
    fiber_nu23: float | None = None,
    fiber_g12: float | None = None,
    fiber_g23: float | None = None,
) -> dict[str, float]:
    """Return effective elastic constants for a short-fiber composite.

    The function applies Mori-Tanaka homogenization to a two-phase composite
    with an isotropic matrix and identical prolate spheroidal fibers, followed
    by Advani-Tucker orientation averaging. All modulus inputs must use one
    consistent unit; returned moduli use that same unit.

    Parameters
    ----------
    fiber_e1 : float
        Fiber axial Young's modulus, greater than zero. For an isotropic
        fiber this is its Young's modulus. For example, ``230`` when all
        moduli are supplied in GPa.
    fiber_nu12 : float
        Fiber axial Poisson's ratio, dimensionless. For an isotropic fiber it
        must be in ``(-1, 0.5)``. For a transversely isotropic fiber it must,
        together with the other constants, produce a positive-definite
        compliance matrix.
    matrix_e : float
        Isotropic-matrix Young's modulus, greater than zero, in the same unit
        as every fiber modulus. For example, ``3.8`` GPa.
    matrix_nu : float
        Isotropic-matrix Poisson's ratio, dimensionless and in ``(-1, 0.5)``.
    fiber_volume_fraction : float
        Fiber volume fraction, dimensionless and in ``[0, 1]``.
    aspect_ratio : float
        Fiber length-axis to transverse-axis ratio, dimensionless, finite,
        and at least ``1``. A value of ``1`` is a sphere; values above ``1``
        describe prolate spheroids. Oblate inclusions are not supported.
    orientation_a11 : float
        Global 11 component of the second-order fiber orientation tensor
        ``a_ij = <p_i p_j>``, dimensionless. For example, ``0.7``.
    orientation_a22 : float
        Global 22 component of the second-order fiber orientation tensor,
        dimensionless. The 33 component is inferred as
        ``1 - orientation_a11 - orientation_a22``.
    orientation_a12 : float, optional
        Global 12 component of the symmetric orientation tensor,
        dimensionless. The complete inferred tensor must be positive
        semidefinite and have unit trace.
    orientation_a13 : float, optional
        Global 13 component of the symmetric orientation tensor,
        dimensionless, subject to the same tensor admissibility conditions.
    orientation_a23 : float, optional
        Global 23 component of the symmetric orientation tensor,
        dimensionless, subject to the same tensor admissibility conditions.
    closure : {"linear", "quadratic", "hybrid", "orthotropic"}, optional
        Approximation used to estimate the fourth-order orientation tensor
        from its second-order tensor. The default is ``"linear"``.
    fiber_isotropic : bool, optional
        Select the constituent model. If ``True`` (the default), construct an
        isotropic fiber from ``fiber_e1`` and ``fiber_nu12`` and ignore the
        four optional transverse-fiber arguments. If ``False``, all four of
        those arguments are required.
    fiber_e2 : float, optional
        Transverse Young's modulus for a transversely isotropic fiber, greater
        than zero and in the same unit as ``fiber_e1``.
    fiber_nu23 : float, optional
        Transverse-plane Poisson's ratio for a transversely isotropic fiber,
        dimensionless and in ``(-1, 1)``.
    fiber_g12 : float, optional
        Axial shear modulus for a transversely isotropic fiber, greater than
        zero and in the same modulus unit.
    fiber_g23 : float, optional
        Transverse shear modulus for a transversely isotropic fiber, greater
        than zero and in the same modulus unit. It must agree within 1% with
        ``fiber_e2 / (2 * (1 + fiber_nu23))``.

    Returns
    -------
    dict[str, float]
        Effective global Young's moduli ``E1``, ``E2``, and ``E3``; Poisson's
        ratios ``nu12``, ``nu13``, and ``nu23``; and engineering shear moduli
        ``G12``, ``G13``, and ``G23``. Young's and shear moduli use the input
        modulus unit. Poisson's ratios are dimensionless.

    Raises
    ------
    ValueError
        If a required value is missing or nonfinite; a modulus is not
        positive; a Poisson's ratio, volume fraction, aspect ratio, closure,
        or orientation tensor is inadmissible; transverse fiber constants are
        inconsistent; or the resulting stiffness matrix is not symmetric and
        positive definite.
    """
    if not isinstance(fiber_isotropic, bool):
        raise ValueError("fiber_isotropic must be true or false")

    if fiber_isotropic:
        fiber_stiffness = core.isotropicC(fiber_e1, fiber_nu12, "fiber")
    else:
        fiber_stiffness = _transversely_isotropic_fiber_stiffness(
            fiber_e1,
            fiber_e2,
            fiber_nu12,
            fiber_nu23,
            fiber_g12,
            fiber_g23,
        )

    matrix_stiffness = core.isotropicC(matrix_e, matrix_nu, "matrix")
    core.validate_volume_fraction(fiber_volume_fraction)

    eshelby_tensor = core.eshelby(matrix_nu, aspect_ratio)
    concentration = core.strainConcentration(
        fiber_volume_fraction,
        eshelby_tensor,
        fiber_stiffness,
        matrix_stiffness,
    )
    unidirectional_stiffness = matrix_stiffness + fiber_volume_fraction * (
        fiber_stiffness - matrix_stiffness
    ) @ concentration

    orientation_a33 = 1.0 - orientation_a11 - orientation_a22
    orientation = core.np.array(
        [
            [orientation_a11, orientation_a12, orientation_a13],
            [orientation_a12, orientation_a22, orientation_a23],
            [orientation_a13, orientation_a23, orientation_a33],
        ],
        dtype=float,
    )
    core.validate_orientation_tensor(orientation)

    if orientation_a33**2 < 1.0e-4:
        linear_c1, linear_c2 = -1.0 / 24.0, 1.0 / 6.0
        hybrid_a, hybrid_b = 2.0, 1.0
    else:
        linear_c1, linear_c2 = -1.0 / 35.0, 1.0 / 7.0
        hybrid_a, hybrid_b = 1.5, 0.5

    if not isinstance(closure, str):
        raise ValueError(
            "closure must be linear, quadratic, hybrid, or orthotropic"
        )
    closure_name = closure.strip().lower()
    if closure_name == "linear":
        fourth_order = core.linearClosure(
            orientation, linear_c1, linear_c2
        )
    elif closure_name == "quadratic":
        fourth_order = core.quadraticClosure(orientation)
    elif closure_name == "hybrid":
        fourth_order = core.hybridClosure(
            orientation,
            hybrid_a,
            hybrid_b,
            linear_c1,
            linear_c2,
        )
    elif closure_name == "orthotropic":
        fourth_order = core.orthoClosure(orientation)
    else:
        raise ValueError(
            "closure must be linear, quadratic, hybrid, or orthotropic"
        )

    averaged_stiffness = core.ornavg(
        unidirectional_stiffness, fourth_order, orientation
    )
    core.validate_stiffness_matrix(
        averaged_stiffness, "homogenized stiffness"
    )
    compliance = core.np.linalg.inv(averaged_stiffness)

    return {
        "E1": float(1.0 / compliance[0, 0]),
        "E2": float(1.0 / compliance[1, 1]),
        "E3": float(1.0 / compliance[2, 2]),
        "nu12": float(-compliance[0, 1] / compliance[0, 0]),
        "nu13": float(-compliance[0, 2] / compliance[0, 0]),
        "nu23": float(-compliance[1, 2] / compliance[1, 1]),
        "G12": float(1.0 / compliance[5, 5]),
        "G13": float(1.0 / compliance[4, 4]),
        "G23": float(1.0 / compliance[3, 3]),
    }
