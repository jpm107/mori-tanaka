# Mori-Tanaka homogenization for short-fiber composites

This package calculates the effective linear-elastic engineering constants of
a two-phase short-fiber composite. It uses Mori-Tanaka mean-field
homogenization for identical prolate spheroidal inclusions and Advani-Tucker
orientation averaging to account for a supplied second-order fiber orientation
tensor.

CompositesAI should serve the single public function
`mori_tanaka.effective_properties`.

## Usage

Supply every modulus in one consistent unit. The returned Young's and shear
moduli use that same unit; Poisson's ratios, volume fraction, aspect ratio, and
orientation components are dimensionless. This example uses GPa:

```python
from mori_tanaka import effective_properties

result = effective_properties(
    fiber_e1=230.0,
    fiber_nu12=0.2,
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

for name, value in result.items():
    print(f"{name}: {value:.6g}")
```

Output:

```text
E1: 18.2068
E2: 8.03415
E3: 7.20532
nu12: 0.416651
nu13: 0.388421
nu23: 0.500619
G12: 3.74156
G13: 2.92794
G23: 2.43433
```

In this example, `E1`, `E2`, `E3`, `G12`, `G13`, and `G23` are in GPa.

## Inputs and conventions

The matrix must be isotropic and is defined by `matrix_e` and `matrix_nu`.
Fibers are isotropic by default and are defined by `fiber_e1` and
`fiber_nu12`.

For a transversely isotropic fiber, set `fiber_isotropic=False` and also
supply `fiber_e2`, `fiber_nu23`, `fiber_g12`, and `fiber_g23`. The transverse
constants must satisfy
`fiber_g23 = fiber_e2 / (2 * (1 + fiber_nu23))` within 1%.

The second-order orientation tensor is

```text
    [a11  a12  a13]
a = [a12  a22  a23]
    [a13  a23  a33]
```

where `a33 = 1 - a11 - a22`. It represents
`a_ij = <p_i p_j>`, where `p` is a unit vector along a fiber in the global
coordinate system. The tensor must be symmetric, positive semidefinite, and
have unit trace.

The `closure` argument estimates the fourth-order orientation tensor from this
second-order tensor:

- `"linear"`: linear closure, generally intended for nearly random states;
- `"quadratic"`: quadratic closure, generally intended for strongly aligned
  states;
- `"hybrid"`: interpolation between the linear and quadratic closures;
- `"orthotropic"`: fitted orthotropic closure (ORW formulation).

Internally, stiffness and compliance use engineering Voigt order
`[11, 22, 33, 23, 13, 12]`, with engineering shear strains
`[2e23, 2e13, 2e12]`.

## Applicability and limitations

- Linear, small-strain elasticity only.
- Two phases: one isotropic matrix and one population of identical fibers.
- Isotropic or transversely isotropic elastic fibers only.
- Prolate spheroidal inclusions with `aspect_ratio >= 1`; oblate inclusions are
  not supported.
- Orientation must be represented by a physically admissible second-order
  tensor and one of the provided closure approximations.
- The package does not model nonlinear response, plasticity, viscoelasticity,
  damage, interfacial debonding, porosity, fiber-length distributions, or
  multiple inclusion families.
- Returned values are homogenized effective constants, not constituent-level
  stresses or local fields.

Invalid or physically inadmissible input raises `ValueError` with a message
identifying the value or relationship that must be corrected.

## Theory

The implementation combines the Mori-Tanaka/Benveniste mean-field formulation
with the Eshelby solution for a spheroidal inclusion. Orientation averaging
uses the tensor representation introduced by Advani and Tucker. The
`"orthotropic"` option uses the fitted orthotropic closure implemented in the
original tool.

This package modernizes a 2018 Mori-Tanaka tool copyrighted by Nicholas Smith.
The original copyright and GNU General Public License version 3 terms are
preserved in `LICENSE.txt`.
