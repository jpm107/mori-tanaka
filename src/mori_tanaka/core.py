"""Numerical building blocks for Mori-Tanaka homogenization."""

import numpy as np

#TODO: improve function documentation

# Voigt convention used throughout this module:
#   stress = [s11, s22, s33, s23, s13, s12]
#   strain = [e11, e22, e33, 2*e23, 2*e13, 2*e12]
# Stiffness and compliance matrices therefore contain G and 1/G,
# respectively, in uncoupled shear entries.
ENGINEERING_STRAIN_SCALE = np.diag([1., 1., 1., 2., 2., 2.])


def _finite_scalar(value, name):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError("%s must be a finite scalar" % name)


def validate_isotropic_material(E, nu, name="isotropic material"):
    """Validate the stability conditions for isotropic linear elasticity."""
    _finite_scalar(E, "%s Young's modulus" % name)
    _finite_scalar(nu, "%s Poisson's ratio" % name)
    if E <= 0.:
        raise ValueError("%s Young's modulus must be positive" % name)
    if not -1. < nu < 0.5:
        raise ValueError(
            "%s Poisson's ratio must satisfy -1 < nu < 0.5" % name
        )


def validate_volume_fraction(vf):
    _finite_scalar(vf, "fiber volume fraction")
    if not 0. <= vf <= 1.:
        raise ValueError("fiber volume fraction must be between 0 and 1")


def validate_orientation_tensor(a2, tolerance=1.e-10):
    """Require a symmetric, unit-trace, positive-semidefinite second moment."""
    a2 = np.asarray(a2, dtype=float)
    if a2.shape != (3, 3):
        raise ValueError("orientation tensor must be a 3x3 matrix")
    if not np.all(np.isfinite(a2)):
        raise ValueError("orientation tensor must contain only finite values")
    if not np.allclose(a2, a2.T, rtol=0., atol=tolerance):
        raise ValueError("orientation tensor must be symmetric")
    if not np.isclose(np.trace(a2), 1., rtol=0., atol=tolerance):
        raise ValueError("orientation tensor must have trace 1")
    eigenvalues = np.linalg.eigvalsh(a2)
    if eigenvalues[0] < -tolerance:
        raise ValueError(
            "orientation tensor must be positive semidefinite; eigenvalues are %s"
            % eigenvalues
        )


def validate_stiffness_matrix(C, name="stiffness matrix"):
    """Require a finite, symmetric, positive-definite engineering stiffness."""
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("%s must be a 6x6 matrix" % name)
    if not np.all(np.isfinite(C)):
        raise ValueError("%s must contain only finite values" % name)
    if not np.allclose(C, C.T, rtol=1.e-10, atol=1.e-8):
        raise ValueError("%s must be symmetric" % name)
    try:
        np.linalg.cholesky(0.5*(C + C.T))
    except np.linalg.LinAlgError:
        raise ValueError("%s must be positive definite" % name)


def validate_matrix_model(is_isotropic):
    if not is_isotropic:
        raise ValueError(
            "the implemented spheroidal Eshelby tensor requires an isotropic "
            "matrix; anisotropic fiber properties remain supported"
        )


def validate_transverse_isotropy(E_transverse, nu_transverse, G_transverse,
                                  name="transversely isotropic material",
                                  relative_tolerance=0.01):
    """Check isotropy of the plane transverse to the material symmetry axis."""
    _finite_scalar(E_transverse, "%s transverse Young's modulus" % name)
    _finite_scalar(nu_transverse, "%s transverse Poisson's ratio" % name)
    _finite_scalar(G_transverse, "%s transverse shear modulus" % name)
    if E_transverse <= 0.:
        raise ValueError("%s transverse Young's modulus must be positive" % name)
    if G_transverse <= 0.:
        raise ValueError("%s transverse shear modulus must be positive" % name)
    if not -1. < nu_transverse < 1.:
        raise ValueError(
            "%s transverse Poisson's ratio must satisfy -1 < nu23 < 1" % name
        )
    if relative_tolerance < 0.:
        raise ValueError("relative tolerance must be nonnegative")

    required_G = E_transverse/(2.*(1.+nu_transverse))
    relative_difference = abs(G_transverse-required_G)/required_G
    if relative_difference > relative_tolerance:
        raise ValueError(
            "%s properties are inconsistent with transverse isotropy: "
            "supplied G23=%g Pa, but E2/[2*(1+nu23)]=%g Pa "
            "(%g%% difference; allowed %g%%). Check E2, nu23, and G23."
            % (name, G_transverse, required_G, 100.*relative_difference,
               100.*relative_tolerance)
        )

#rotation functions
def qij_x(theta):
    """
    rotation tensor about x-axis by some theta
    input: theta (angle in radians)
    output: qij (3x3 rotation tensor)
    """
    qij = np.array([[1,0,0],
                   [0,np.cos(theta),np.sin(theta)],
                   [0,-np.sin(theta),np.cos(theta)]])
    return qij

def qij_y(theta):
    """
    rotation tensor abo|ut y-axis by some theta
    input: theta (angle in radians)
    output: qij (3x3 rotation tensor)
    """
    qij = np.array([[np.cos(theta), 0, -np.sin(theta)],
                   [0,1,0],
                   [np.sin(theta),0,np.cos(theta)]])
    return qij

def qij_z(theta):
    """
    rotation tensor about z-axis by some theta
    input: theta (angle in radians)
    output: qij (3x3 rotation tensor)
    """
    qij = np.array([[np.cos(theta),np.sin(theta),0],
                   [-np.sin(theta),np.cos(theta),0],
                   [0,0,1]])
    return qij

def R_sigma(q):
    """
    rotation matrix for engineering notation given some rotation tensor, qij
    note: uses convention for sigma = [s11, s22, s33, s23, s13, s12]
    input: q (3x3 rotation tensor)
    output: R_s (6x6 rotation matrix)
    """
    R_s = np.array([[q[0,0]**2,q[0,1]**2,q[0,2]**2,2.*q[0,1]*q[0,2],2.*q[0,0]*q[0,2], 2.*q[0,0]*q[0,1]],
                       [q[1,0]**2,q[1,1]**2,q[1,2]**2,2.*q[1,1]*q[1,2],2.*q[1,0]*q[1,2], 2.*q[1,0]*q[1,1]],
                       [q[2,0]**2,q[2,1]**2,q[2,2]**2,2.*q[2,1]*q[2,2],2.*q[2,0]*q[2,2], 2.*q[2,0]*q[2,1]],
                       [q[1,0]*q[2,0], q[1,1]*q[2,1], q[1,2]*q[2,2],q[1,2]*q[2,1]+q[1,1]*q[2,2], q[1,2]*q[2,0]+q[1,0]*q[2,2], q[1,1]*q[2,0]+q[1,0]*q[2,1]],
                       [q[0,0]*q[2,0], q[0,1]*q[2,1], q[0,2]*q[2,2],q[0,2]*q[2,1]+q[0,1]*q[2,2], q[0,2]*q[2,0]+q[0,0]*q[2,2], q[0,1]*q[2,0]+q[0,0]*q[2,1]],
                       [q[0,0]*q[1,0], q[0,1]*q[1,1], q[0,2]*q[1,2],q[0,2]*q[1,1]+q[0,1]*q[1,2], q[0,2]*q[1,0]+q[0,0]*q[1,2], q[0,1]*q[1,0]+q[0,0]*q[1,1]]])
    return R_s

# for printing purposes, recast 4th-order tensor as 6x6
def tens2eng(tens):
    """
    returns 4th-order tensor in engineering notation
    """
    A = np.zeros((6,6))
    #mapping
    code = {0:(0,0),1:(1,1),2:(2,2),3:(1,2),4:(0,2),5:(0,1)}
    for i in range(6):
        for j in range(6):
            A[i,j] = tens[code[i][0],code[i][1],code[j][0],code[j][1]]
    return A

#Eshelby tensor calculations
def _prolate_shape_terms(s, Q):
    """Return I1 and T for a prolate spheroid, including the sphere limit."""
    if not np.isfinite(s) or s < 1.:
        raise ValueError("aspect ratio s must be finite and at least 1")

    u = s**2. - 1.
    if u < 1.e-3:
        # Taylor expansions of the exact prolate-spheroid expressions about
        # u = s**2 - 1 = 0. They provide the analytic spherical limit and
        # avoid cancellation between nearly equal terms close to s = 1.
        I1 = 4./3. + 4.*u/15. - 16.*u**2/105. + 32.*u**3/315.
        T = Q*(-4./15. + 16.*u/105. - 32.*u**2/315.)
    else:
        I1 = (2.*s/u**1.5)*(s*u**.5-np.arccosh(s))
        T = Q*(4.-3.*I1)/(3.*u)
    return I1, T


def eshelby(nu_m, s):
    Q = 3./(8.*(1-nu_m))
    R = (1.-2.*nu_m)/(8.*(1.-nu_m))
    I1, T = _prolate_shape_terms(s, Q)
    I3 = 4.-2.*I1
    S = np.zeros((6,6))
    S[0,0] = Q + R*I1 + 0.75*T
    S[1,1] = S[0,0]
    S[2,2] = 4./3.*Q + R*I3 + 2.*s**2.*T
    # Transverse cross-component. This is the Q-R-T form of the published
    # spheroidal S_1122 expression (after relabelling the long axis to axis 3).
    S[0,1] = Q/3. - R*I1 + T/4.
    S[1,0] = S[0,1]
    S[0,2] = -R*I1 - s**2*T
    S[1,2] = S[0,2]
    S[2,0] = -R*I3 - T
    S[2,1] = S[2,0]
    S[5,5] = Q/3 + R*I1 + T/4
    S[3,3] = 2*R - I1*R/2 - (1+s**2)*T/2
    S[4,4] = S[3,3]
    # Keep the inclusion axis in the local 3-direction.  The orientation
    # averaging coefficients in ornavg() use this as their reference axis.
    # The formulas above give the tensor components S_ijkl. Convert them to
    # the operator that maps engineering strain vectors to engineering strain
    # vectors; this doubles the three shear-output rows.
    return np.dot(ENGINEERING_STRAIN_SCALE,S)

#fiber/matrix stiffness matrices
def isotropicC(E,nu,name="isotropic material"):
    validate_isotropic_material(E, nu, name)
    C = np.zeros((6,6))
    C[0:3,0:3] = nu*np.ones((3,3))
    C = C + (1.-2.*nu)*np.eye(6)
    C[3:,3:] = C[3:,3:]/2.
    C = E/((1.+nu)*(1.-2.*nu))*C
    return C

#orthotropic stiffness matrices
def orthotropicC(E1,E2,E3,vxy,vyz,vxz,Gxy,Gyz,Gxz,
                 name="orthotropic material"):
    moduli = (E1, E2, E3, Gxy, Gyz, Gxz)
    modulus_names = ("E1", "E2", "E3", "G12", "G23", "G13")
    for value, modulus_name in zip(moduli, modulus_names):
        _finite_scalar(value, "%s %s" % (name, modulus_name))
        if value <= 0.:
            raise ValueError("%s %s must be positive" % (name, modulus_name))
    for value, poisson_name in zip((vxy, vyz, vxz), ("nu12", "nu23", "nu13")):
        _finite_scalar(value, "%s %s" % (name, poisson_name))
    S = np.array([[1./E1, -vxy/E1, -vxz/E1, 0, 0, 0],
              [-vxy/E1, 1./E2, -vyz/E2, 0, 0, 0],
              [-vxz/E1, -vyz/E2, 1./E3, 0, 0, 0],
              [0, 0, 0, 1./Gyz, 0, 0],
              [0, 0, 0, 0, 1./Gxz, 0],
              [0, 0, 0, 0, 0, 1./Gxy]])
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        raise ValueError("%s compliance must be positive definite" % name)
    C = np.linalg.inv(S)
    validate_stiffness_matrix(C, "%s stiffness" % name)
    return C

#Eshelby strain concentration
def strainConcentration(vf,S,Cf,Cm):
    validate_volume_fraction(vf)
    validate_stiffness_matrix(Cf, "fiber stiffness")
    validate_stiffness_matrix(Cm, "matrix stiffness")
    if np.shape(S) != (6, 6) or not np.all(np.isfinite(S)):
        raise ValueError("Eshelby tensor must be a finite 6x6 matrix")
    A_e = np.linalg.inv(np.eye(6)+np.dot(np.dot(S,np.linalg.inv(Cm)),Cf-Cm))
    A_mt = np.dot(A_e,np.linalg.inv(vf*A_e+(1-vf)*np.eye(6)))
    return A_mt

#orientation averaging
def ornavg(C,A4,a2):
    B1 = C[0,0] + C[2,2] - 2*C[1,2] -4*C[4,4]
    B2 = C[1,2] - C[0,1]
    B3 = C[4,4] - 0.5*(C[0,0]-C[0,1])
    B4 = C[0,1]
    B5 = 0.5*(C[0,0]-C[0,1])
    #map between tensor and engineering notation
    code = {0:(0,0),1:(1,1),2:(2,2),3:(1,2),4:(0,2),5:(0,1)}
    C_avg = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            C_avg[i,j] = B1*A4[i,j] + B2*(
            a2[code[i][0],code[i][1]]*(code[j][0]==code[j][1]) +
                a2[code[j][0],code[j][1]]*(code[i][0]==code[i][1])) + B3*(
            a2[code[i][0],code[j][0]]*(code[i][1]==code[j][1]) +
            a2[code[i][0],code[j][1]]*(code[i][1]==code[j][0]) +
            a2[code[i][1],code[j][0]]*(code[i][0]==code[j][1]) +
            a2[code[i][1],code[j][1]]*(code[i][0]==code[j][0])) + B4*(
            (code[i][0]==code[i][1])*(code[j][0]==code[j][1])) + B5*(
            (code[i][0]==code[j][0])*(code[i][1]==code[j][1]) +
            (code[i][0]==code[j][1])*(code[i][1]==code[j][0]))
    return C_avg

def linearClosure(a2,c1,c2):
    validate_orientation_tensor(a2)
    a4 = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    a4[i,j,k,l] = c1*((i==j)*(k==l) + (i==k)*(j==l) + (i==l)*(j==k)) + c2*(
                     a2[i,j]*(k==l) + a2[i,k]*(j==l) +a2[i,l]*(j==k) + a2[k,l]*(i==j) +
                     a2[j,l]*(i==k) + a2[j,k]*(i==l))
    return tens2eng(a4)

def quadraticClosure(a2):
    validate_orientation_tensor(a2)
    a4 = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    a4[i,j,k,l] = a2[i,j]*a2[k,l]
    return tens2eng(a4)

def hybridClosure(a2,A,B,c1,c2):
    validate_orientation_tensor(a2)
    f = -B
    for i in range(3):
        for j in range(3):
            f = f + A*a2[i,j]*a2[j,i]
    a4_l = linearClosure(a2,c1,c2)
    a4_q = quadraticClosure(a2)
    a4 = (1-f)*a4_l + f*a4_q
    return a4

#TODO: fitted orthotropic closure
def orthoClosure(a2):
    validate_orientation_tensor(a2)
    #based on https://onlinelibrary.wiley.com/doi/epdf/10.1002/pc.10566
    #ORW, TODO: ORW3
    # The ORW fit defines a1 >= a2 >= a3 and uses the two largest
    # eigenvalues as its polynomial variables. eigh() is appropriate for the
    # symmetric orientation tensor, but returns ascending eigenvalues, so the
    # eigenvalues and their corresponding eigenvector columns must be reversed.
    ae, q = np.linalg.eigh(a2)
    order = np.argsort(ae)[::-1]
    ae = ae[order]
    q = q[:,order]
    # Use a consistent proper-orthogonal principal basis. Eigenvector signs do
    # not affect this even-order tensor, but this removes an arbitrary choice.
    if np.linalg.det(q) < 0.:
        q[:,-1] *= -1.
    #estimate unknown terms in orthotropic a4 tensor
    C = np.array([[0.070055, 0.339376, 0.590331, -0.396796, 0.333693, 0.411944],
        [0.115177, -0.368267, 0.252880, 0.094820, 0.800181, 0.535224],
        [1.249811, -2.148297, 0.898521, -2.290157, 1.044147, 1.934914]])
    a = np.array([1,ae[0],ae[0]**2,ae[1],ae[1]**2,ae[0]*ae[1]])
    ab = np.dot(C,a)
    #determine other terms in a4
    x = np.array([[1,1,0],
        [1,0,1],
        [0,1,1]])
    y = np.array([ae[0]-ab[0],ae[1]-ab[1],ae[2]-ab[2]])
    ax = np.linalg.solve(x,y)
    A4 = np.array([[ab[0],ax[0],ax[1],0,0,0],
                    [ax[0],ab[1],ax[2],0,0,0],
                    [ax[1],ax[2],ab[2],0,0,0],
                    [0,0,0,ax[2],0,0],
                    [0,0,0,0,ax[1],0],
                    [0,0,0,0,0,ax[0]]])
    degeneracy_tolerance = 1.e-10
    if ae[0]-ae[2] <= degeneracy_tolerance:
        # With no preferred principal direction, objectivity and the
        # contraction A_ijkk=a_ij require the isotropic fourth moment.
        A4 = np.zeros((6,6))
        A4[0:3,0:3] = 1./15.
        np.fill_diagonal(A4[0:3,0:3],1./5.)
        np.fill_diagonal(A4[3:,3:],1./15.)
    elif ae[0]-ae[1] <= degeneracy_tolerance:
        # Average over rotations in the repeated 1-2 eigenspace. This is the
        # transversely isotropic projection of the fitted fourth moment.
        plane_sum = A4[0,0] + A4[1,1] + 2.*A4[0,1]
        plane_normal = 3.*plane_sum/8.
        plane_cross = plane_sum/8.
        axial_cross = 0.5*(A4[0,2] + A4[1,2])
        A4[0,0] = plane_normal
        A4[1,1] = plane_normal
        A4[0,1] = A4[1,0] = plane_cross
        A4[5,5] = plane_cross
        A4[0,2] = A4[2,0] = axial_cross
        A4[1,2] = A4[2,1] = axial_cross
        A4[3,3] = axial_cross
        A4[4,4] = axial_cross
    elif ae[1]-ae[2] <= degeneracy_tolerance:
        # Average over rotations in the repeated 2-3 eigenspace.
        plane_sum = A4[1,1] + A4[2,2] + 2.*A4[1,2]
        plane_normal = 3.*plane_sum/8.
        plane_cross = plane_sum/8.
        axial_cross = 0.5*(A4[0,1] + A4[0,2])
        A4[1,1] = plane_normal
        A4[2,2] = plane_normal
        A4[1,2] = A4[2,1] = plane_cross
        A4[3,3] = plane_cross
        A4[0,1] = A4[1,0] = axial_cross
        A4[0,2] = A4[2,0] = axial_cross
        A4[4,4] = axial_cross
        A4[5,5] = axial_cross
    #rotate A4 back out of principal direction
    R = R_sigma(q)
    return np.dot(R,np.dot(A4,R.T))
