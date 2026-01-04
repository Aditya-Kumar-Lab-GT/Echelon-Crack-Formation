###########################################################################################################################################################################################################################
# This FEniCS code implements the new phase-field model introduced in  Kumar, A., Bourdin, B., Francfort, G.A. and Lopez-Pamies, O., 2020. "Revisiting nucleation in the phase-field approach to brittle fracture". Journal of the Mechanics and Physics of Solids, 104027.
# This new phase-field model is used to solve the 'Brazilian Disk test' in Begostone.
# Output: XDMF files to visualize crack nucleation and propagation 
# Contact Aditya Kumar (akumar355@gatech.edu) for questions.
###########################################################################################################################################################################################################################

from dolfin import *
import numpy as np
import time
import csv
import sys
# from ufl import eq
from ufl import (conditional, gt,  lt, eq, ge, le)


comm = MPI.comm_world 
comm_rank = MPI.rank(comm)

# The following shuts off a deprecation warning for quadrature representation:
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

# Properties
E, nu = 9800, 0.13	#Young's modulus and Poisson's ratio

Gc = 0.091125	#Critical energy release rate

sts, scs = 27, 27*2 #40#77	#Tensile strength and compressive strength 77

lch = 3*Gc*E/8/(sts**2) #take lower, about 0.09375 and 0.114 
eps = 0.4 #epsilon should not be chosen to be too large compared to lch. Typically eps<4*lch should work
h = eps/4

mu, lmbda, kappa = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu))), Constant(E/(3*(1 - 2*nu)))
shs = (2*scs*sts)/(3*(scs - sts))
Wts = 0.5*(sts**2)/E
Whs = 0.5*(shs**2)/kappa
delta = (1 + 3 * h / (8 * eps)) ** (-2) * ((sts + (1 + 2 * np.sqrt(3)) * shs) / ((8 + 3 * np.sqrt(3)) * shs)) * 3 * Gc / (16 * Wts * eps) + (1 + 3 * h / (8 * eps)) ** (-1) * (2 / 5)
'''
# Material properties
E, nu = 100000, 0.2	#Young's modulus and Poisson's ratio
Gc= 1.0	#Critical energy release rate
sts, scs= 50, 1000	#Tensile strength and compressive strength
#Irwin characteristic length
lch=3*Gc*E/8/(sts**2)
#The regularization length
eps=0.2  #epsilon should not be chosen to be too large compared to lch. Typically eps<4*lch should work
delta=23.0
h=0.05

# Problem description
comm = MPI.comm_world 
comm_rank = MPI.rank(comm)

#Geometry of the single edge notch geometry
# ac=0.25  #notch length
Dia = 50  #making use of symmetry
Diaeff=Dia*1 #29 for circular anvil of diameter=1.25*Dia
'''
# Create mesh
# mesh=CircleMesh(comm, Point(0.0,0.0), 2.9, int(Dia/(8*h)))
# domain1 = CompiledSubDomain("abs(x[0])<1.5", lch=lch)
# ir=0
# while ir<1:
	# d_markers = MeshFunction("bool", mesh, 2, False)
	# domain1.mark(d_markers, True)
	# mesh = refine(mesh,d_markers, True)
	# ir+=1

# domain2 = CompiledSubDomain("abs(x[0])<1.0", a=ac, eps=eps, h=h)
# ir=0
# while ir<2:
	# d_markers = MeshFunction("bool", mesh, 2, False)
	# domain2.mark(d_markers, True)
	# mesh = refine(mesh,d_markers, True)
	# ir+=1

mesh=Mesh("3dcrack_large.xml")
# mesh=refine(mesh)
# mesh=refine(mesh)
n=FacetNormal(mesh)

Lx = 25
Ly = 25
Lz = 25
CrackZ = 12.5

# Mark boundary subdomians
left = CompiledSubDomain("x[0]<1e-4")
right = CompiledSubDomain("x[0]>Lx-1e-4", Lx=Lx)
top = CompiledSubDomain("x[1]>Ly-1e-4", Ly=Ly)
bottom = CompiledSubDomain("x[1]<1e-4")
outer= CompiledSubDomain("x[1]<.3*Ly or x[1]>.7*Ly", Ly=Ly)
crack = CompiledSubDomain("abs(x[1]-.5*Ly)<2*h && x[0]<CrackZ+2*h && x[0]>CrackZ-2*h", Ly=Ly, CrackZ=CrackZ, h=h)


set_log_level(40)  #Error level=40, warning level=30
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


# Choose phase-field model
phase_model=1; 
filemodel='KLPI1scs2'

# Define function space
V = VectorFunctionSpace(mesh, "CG", 1)   #Function space for u
Y = FunctionSpace(mesh, "CG", 1)         #Function space for z
V_stress = TensorFunctionSpace(mesh, "CG", 1)

disp=0.25 #1% strain rate
r = Expression("t*disp",degree=1,t=0,disp=disp)
ry = Expression("t*disp*0.0",degree=1,t=0,disp=disp)
r0= Expression("(t-tau)*disp",degree=1,t=0,disp=disp, tau=0)

# Define Dirichlet boundary conditions
c=Expression("0.0",degree=1,t=0)
								
bct= DirichletBC(V.sub(2), r, top)
bct2= DirichletBC(V.sub(1), c, top)
bct3= DirichletBC(V.sub(0), c, top)
bcb= DirichletBC(V.sub(0), c, bottom)
bcb2= DirichletBC(V.sub(1), c, bottom)
bcb3= DirichletBC(V.sub(2), c, bottom)
bcs = [bct, bct2, bct3, bcb, bcb2, bcb3]

bct_du0= DirichletBC(V.sub(2), r0, top)
bct_du= DirichletBC(V.sub(2), c, top)
bcs_du0 = [bct_du0, bct2, bct3, bcb, bcb2, bcb3]
bcs_du = [bct_du, bct2, bct3, bcb, bcb2, bcb3]

cz=Constant(1.0)
cz2=Constant(0.0)
bct_z = DirichletBC(Y, cz, outer)
bct_z2 = DirichletBC(Y, cz2, crack)
# cz2=Constant(0.0)
# bct_z2 = DirichletBC(Y, cz2, cracktip)
bcs_z=[bct_z]

bct_dz = DirichletBC(Y, Constant(0.0), outer)
bct_dz2 = DirichletBC(Y, Constant(0.0), crack)
bcs_dz=[bct_dz]

boundary_subdomains = MeshFunction("size_t", mesh, 1)
boundary_subdomains.set_all(0)
left.mark(boundary_subdomains,1)
right.mark(boundary_subdomains,1)
bottom.mark(boundary_subdomains,2)
top.mark(boundary_subdomains,3)
ds = ds(subdomain_data=boundary_subdomains)

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
u_inc = Function(V)
dz = TrialFunction(Y)            # Incremental phase field
y  = TestFunction(Y)             # Test function
z  = Function(Y)                 # Phase field from previous iteration
z_inc = Function(Y)
d = u.geometric_dimension()


#Initialisation of displacement field,u and the phase field,z
u_init = Constant((0.0,  0.0, 0.0))
u.interpolate(u_init)
for bc in bcs:
	bc.apply(u.vector())

z_init = Constant(1.0)
z.interpolate(z_init)
for bc in bcs_z:
	bc.apply(z.vector())

z_ub = Function(Y)
z_ub.interpolate(Constant(1.0))	
z_lb = Function(Y)
z_lb.interpolate(Constant(-0.0))
	

u_prev = Function(V)
assign(u_prev,u)
z_prev = Function(Y)
assign(z_prev,z)


	
#Label the dofs on boundary
def extract_dofs_boundary(V, bsubd):	
	label = Function(V)
	label_bc_bsubd = DirichletBC(V, Constant((1,1,1)), bsubd, method='pointwise'   )
	label_bc_bsubd.apply(label.vector())
	bsubd_dofs = np.where(label.vector()==1)[0]
	return bsubd_dofs

#Dofs on which reaction is calculated
top_dofs=extract_dofs_boundary(V,top)
y_dofs_top=top_dofs[2::d]


#Function to evaluate a field at a pint
def evaluate_function(u, x):
	comm = u.function_space().mesh().mpi_comm()
	if comm.size == 1:
		return u(*x)

	# Find whether the point lies on the partition of the mesh local
	# to this process, and evaulate u(x)
	cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
	u_eval = u(*x) if distance < DOLFIN_EPS else None

	# Gather the results on process 0
	comm = mesh.mpi_comm()
	computed_u = comm.gather(u_eval, root=0)

	# Verify the results on process 0 to ensure we see the same value
	# on a process boundary
	if comm.rank == 0:
		global_u_evals = np.array([y for y in computed_u if y is not None], dtype=np.double)
		assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)

		computed_u = global_u_evals[0]
	else:
		computed_u = None

	# Broadcast the verified result to all processes
	computed_u = comm.bcast(computed_u, root=0)

	return computed_u

def local_project(v, V, u):
	parameters["form_compiler"]["representation"] = 'quadrature'
	dv = TrialFunction(V)
	v_ = TestFunction(V)
	a_proj = inner(dv, v_)*dx
	b_proj = inner(v, v_)*dx
	solver = LocalSolver(a_proj, b_proj)
	solver.factorize()
	solver.solve_local_rhs(u)


def safe_sqrt(x):
	# A safe square root that avoids negative arguments
	return sqrt(x + 1.0e-16)

def compute_eigenvalues(A, problem_dim=None):
    
	if problem_dim == 2:
		# 2D case: closed-form solution for a 2x2 tensor
		eig1 = (tr(A) + sqrt(abs(tr(A)**2 - 4 * det(A)))) / 2
		eig2 = (tr(A) - sqrt(abs(tr(A)**2 - 4 * det(A)))) / 2 
		return eig1, eig2

	elif problem_dim == 3:
		# 3D case: analytical solution based on invariants.
		Id = Identity(3)
		I1 = tr(A)
		I2 = (tr(A)**2 - tr(A*A)) / 2.0  # second invariant (not used here)
		I3 = det(A)                     # third invariant
		
		# Parameters for eigenvalue computation
		d_par = I1 / 3.0
		e_par = safe_sqrt(tr((A - d_par*Id)*(A - d_par*Id)) / 6.0)
		
		# Define f_par carefully to avoid division by zero
		zero = 0 * Id
		f_par_expr = (1.0 / e_par) * (A - d_par * Id)
		f_par =  conditional(eq(e_par, 0), zero, f_par_expr)
		
		# Compute the argument of the acos, and bound it to [-1, 1]
		g_par0 = det(f_par) / 2.0
		tol = Constant(3.0e-16)  # tolerance to avoid acos issues at the boundaries
		g_par1 = conditional(ge(g_par0, 1.0 - tol), 1.0 - tol, g_par0)
		g_par = conditional(le(g_par1, -1.0 + tol), -1.0 + tol, g_par1)
		
		# Angle for the eigenvalue formulas
		h_par = acos(g_par) / 3.0
		
		# Compute the eigenvalues (ordered such that eig1 >= eig2 >= eig3)
		eig3 = d_par + 2.0 * e_par * cos(h_par + 2.0 * np.pi / 3.0)
		eig2 = d_par + 2.0 * e_par * cos(h_par + 4.0 * np.pi / 3.0)
		eig1 = d_par + 2.0 * e_par * cos(h_par + 6.0 * np.pi / 3.0)
		return eig1, eig2, eig3

	else:
		raise ValueError("Dimension not supported. Only 2D and 3D cases are implemented.")




##Strain Energy, strain and stress functions in linear isotropic elasticity

I = Identity(3)             # Identity tensor
F = I + grad(u)             # Deformation gradient
J  = det(F)



def energy(v):
	return mu*(inner(sym(grad(v)),sym(grad(v)))) +  0.5*(lmbda)*(tr(sym(grad(v))))**2 
	#return mu*(inner(sym(grad(v)),sym(grad(v))) )+  0.5*(lmbda)*(tr(sym(grad(v))))**2 #plane strain
	
def epsilon(v):
	return sym(grad(v))

def epsilon_dev(v):
	return epsilon(v)-(1/3)*tr(epsilon(v))*Identity(len(v))

def sigma(v):
	return 2.0*mu*sym(grad(v)) + (lmbda)*tr(sym(grad(v)))*Identity(len(v))
	#return 2.0*mu*sym(grad(v)) + (lmbda)*tr(sym(grad(v)))*Identity(len(v)) #plane strain

def sigmavm(sig,v):
	return sqrt(1/2*(inner(sig-1/3*tr(sig)*Identity(len(v)), sig-1/3*tr(sig)*Identity(len(v)))))
	#return sqrt(1/2*(inner(sig-1/3*(1+nu)*tr(sig)*Identity(len(v)), sig-1/3*(1+nu)*tr(sig)*Identity(len(v))))) #plane strain

def eig_plus(A):
	return (tr(A) + sqrt(abs(tr(A)**2-4*det(A))))/2

def eig_minus(A):
	return (tr(A) - sqrt(abs(tr(A)**2-4*det(A))))/2

'''
eta=1e-7
# Stored strain energy density (compressible L-P model)
psi1 =(z**2+eta)*(energy(u))	
psi11=2*z*energy(u)
# Total potential energy
Pi = psi1*dx #- dot(Tf*n, u)*ds(1)
# Compute first variation of Pi (directional derivative about u in the direction of v)
R = derivative(Pi, u, v) 

# Compute Jacobian of R
Jac = derivative(R, u, du)
Jac3=inner(grad(du),grad(v))*dx

beta0=-3*Gc/8/eps*delta
beta1= ((9*E*Gc)/2. - (9*E*Gc*sts)/(2.*scs))/(24.*E*eps*sts) + (-12*beta0*E + (12*beta0*E*sts)/scs - 12*sts**2)/(24.*E*sts)
beta2= (9*E*Gc*scs + 9*E*Gc*sts)/(16.*sqrt(3)*E*eps*scs*sts) + (-24*beta0*E*scs - 24*beta0*E*sts - 24*scs*sts**2)/(16.*sqrt(3)*E*scs*sts)



pen=1000*conditional(lt(-beta0,Gc/eps),Gc/eps, -beta0)

ce= (beta1*(z**2)*(tr(sigma(u))) + beta2*(z**2)*(sigmavm(sigma(u),u)) +beta0) - (1- abs(tr(sigma(u)))/tr(sigma(u)))*z*((sigmavm(sigma(u),u)**2)/(2*mu) + (tr(sigma(u))**2)/(18*kappa))



f1 = (scs-sts)/(sqrt(3)*(scs + sts))
f0 = -(2*scs*sts)/(sqrt(3)*(scs + sts))
FDP = (sigmavm(sigma(u),u)) + f1*(tr(sigma(u))) + f0
FDP2=2*energy(u)+ ((beta1*(tr(sigma(u))) + beta2*(sigmavm(sigma(u),u)) +beta0) - (1- abs(tr(sigma(u)))/tr(sigma(u)))*((sigmavm(sigma(u),u)**2)/(2*mu) + (tr(sigma(u))**2)/(18*kappa)))-3*Gc/8/eps
indicator=conditional(gt(FDP, 0), 1, 0)
indicator2=conditional(gt(FDP2, 0), 1, 0)
'''

# ---------------------------------------------------------------------------------------------------------------------------------------------
# 1. JMPS2020		Surfing Complete Model
# ---------------------------------------------------------------------------------------------------------------------------------------------
if phase_model == 1:

#        eps = lch/2
	shs = (2*scs*sts)/(3*(scs-sts))
	Wts = 0.5*(sts**2)/E
	Whs = 0.5*(shs**2)/kappa
	delta = (1+3*h/(8*eps))**(-2) * ((sts+(1+2*sqrt(3))*shs)/((8+3*sqrt(3))*shs)) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1)*(2/5)
	
	pen=1000*(3*Gc/8/eps) * conditional(lt(delta, 1), 1, delta)
	eta=1e-7
	
	# Stored strain energy density (compressible L-P model)
	psi1 =(z**2+eta)*(energy(u))	
	psi11=energy(u)
	stress=(z**2+eta)*sigma(u)
	
	# Total potential energy
	Pi = psi1*dx
	
	
	I1_d = (z**2)*tr(sigma(u))
	SQJ2_d = (z**2)*sigmavm(sigma(u),u)
	beta1 = -(delta*Gc)/(shs*8*eps) + (2*Whs)/(3*shs)
	beta2  = -(sqrt(3)*(3*shs-sts)*delta*Gc)/(shs*sts*8*eps) - (2*Whs)/(sqrt(3)*shs) + (2*sqrt(3)*Wts)/(sts)
	triaxiality= I1_d/(sqrt(3)*SQJ2_d)
	ce = beta2*SQJ2_d + beta1*I1_d #+ z*(1-sqrt(I1_d**2)/I1_d)*psi11
	# ce = beta2*SQJ2_d + beta1*I1_d + z*(1-sqrt((triaxiality+0)**2)/(triaxiality+0))*psi11
	# ce = beta2*(SQJ2_d +(12*pho_c/(pho_c-1)**2)*I1_d**2) + beta1*I1_d + z*(1-sqrt((triaxiality+1/2)**2)/(triaxiality+1/2))*psi11

	f1 = (scs-sts)/(sqrt(3)*(scs + sts))
	f0 = -(2*scs*sts)/(sqrt(3)*(scs + sts))
	FDP = (sigmavm(sigma(u),u)) + f1*(tr(sigma(u))) + f0
	FDP2 = 2*psi11 - ce -3*Gc/8/eps
	indicator=conditional(gt(FDP, 0), 1, 0)
	indicator2=conditional(gt(FDP2, 0), 1, 0)
	
	
	# Compute first variation of Pi (directional derivative about u in the direction of v)
	R = derivative(Pi, u, v)
	
	# Compute Jacobian of R
	Jac = derivative(R, u, du)
	
	#To use later for memory allocation for these tensors
	A=PETScMatrix()
	b=PETScVector()
	
	I1_correction_factor = conditional(ge(I1_d, 0), 2, 0)
	
	#Balance of configurational forces PDE
	Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
	Wv2=conditional(le(z, 0.05), 1, 0)*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
	R_z = y*I1_correction_factor*z*(psi11)*dx - y*(ce)*dx + 3*delta*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) +  derivative(Wv2,z,y)   # Complete Model
	
	# Compute Jacobian of R_z
	Jac_z = derivative(R_z, z, dz)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 2. JMPS2020- without I1 correction		Surfing Complete Model
# ---------------------------------------------------------------------------------------------------------------------------------------------

elif phase_model == 2:
#        eps = lch/2
	shs = (2*scs*sts)/(3*(scs-sts))
	Wts = 0.5*(sts**2)/E
	Whs = 0.5*(shs**2)/kappa
	delta = (1+3*h/(8*eps))**(-2) * ((sts+(1+2*sqrt(3))*shs)/((8+3*sqrt(3))*shs)) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1)*(2/5)
	
	pen=1000*(3*Gc/8/eps) * conditional(lt(delta, 1), 1, delta)
	eta=1e-7
	
	# Stored strain energy density (compressible L-P model)
	psi1 =(z**2+eta)*(energy(u))	
	psi11=energy(u)
	stress=(z**2+eta)*sigma(u)
	
	# Total potential energy
	Pi = psi1*dx
	
	
	I1_d = (z**2)*tr(sigma(u))
	SQJ2_d = (z**2)*sigmavm(sigma(u),u)
	beta1 = -(delta*Gc)/(shs*8*eps) + (2*Whs)/(3*shs)
	beta2  = -(sqrt(3)*(3*shs-sts)*delta*Gc)/(shs*sts*8*eps) - (2*Whs)/(sqrt(3)*shs) + (2*sqrt(3)*Wts)/(sts)
	triaxiality= I1_d/(sqrt(3)*SQJ2_d)
	# ce = beta2*SQJ2_d + beta1*I1_d + z*(1-sqrt(I1_d**2)/I1_d)*psi11
	ce = beta2*SQJ2_d + beta1*I1_d 
	# ce = beta2*(SQJ2_d +(12*pho_c/(pho_c-1)**2)*I1_d**2) + beta1*I1_d + z*(1-sqrt((triaxiality+1/2)**2)/(triaxiality+1/2))*psi11

	f1 = (scs-sts)/(sqrt(3)*(scs + sts))
	f0 = -(2*scs*sts)/(sqrt(3)*(scs + sts))
	FDP = (sigmavm(sigma(u),u)) + f1*(tr(sigma(u))) + f0
	FDP2 = 2*psi11 - ce -3*Gc/8/eps
	indicator=conditional(gt(FDP, 0), 1, 0)
	indicator2=conditional(gt(FDP2, 0), 1, 0)
	
	
	# Compute first variation of Pi (directional derivative about u in the direction of v)
	R = derivative(Pi, u, v)
	
	# Compute Jacobian of R
	Jac = derivative(R, u, du)
	
	#To use later for memory allocation for these tensors
	A=PETScMatrix()
	b=PETScVector()
	
	#Balance of configurational forces PDE
	Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
	Wv2=conditional(le(z, 0.05), 1, 0)*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
	R_z = y*2*z*(psi11)*dx - y*(ce)*dx + 3*delta*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) +  derivative(Wv2,z,y)   # Complete Model
	
	# Compute Jacobian of R_z
	Jac_z = derivative(R_z, z, dz)

# ---------------------------------------------------------------------------------------------------------------------------------------------
# 3. PF_CZM 2017	A Unified Phase Field Theory ## Dr Y. Wu
# ---------------------------------------------------------------------------------------------------------------------------------------------

elif phase_model == 3:
#        eps = lch/2
	pen=1000*Gc/eps
	eta=1e-7
	
	pho_c = scs/sts
	a1 = 4*E*Gc/(pi*eps*sts**2)
	a2 = -0.5
	a3 = 0.0
	w_z = z**2/(z**2 + a1*(1-z)*(1+a2*(1-z)*(1+a3*(1-z))))
	stress=(w_z+eta)*sigma(u)
	
	J2 = sigmavm(sigma(u),u)**2
	
	triaxiality= tr(sigma(u))/(sqrt(3*J2))
	sigma_eq = 1/(2*pho_c)*((pho_c-1)*tr(sigma(u)) + sqrt((pho_c-1)**2*(tr(sigma(u)))**2+12*pho_c*J2))		                                        # modified von Mises criterion   # Wu 2020
	#	sigma_eq = (eig_plus(sigma(u))+abs(eig_plus(sigma(u))))/2                                                                                       # Rankine criterion
	y_wu = sigma_eq**2/(2*E)
	Y_wu = -(a1*z*(2+2*a2*(1-z)-z)) / (z**2+a1*(1-z)+a1*a2*(1-z)**2)**2 * y_wu #*(1+sqrt((triaxiality+0)**2)/(triaxiality+0)/2)
	
	
	
	# Stored strain energy density
	psi1 = (w_z+eta)*energy(u)
	#	psi1 = (w_z+eta)*y_wu
	psi11=energy(u)
	
	Energy_diff = psi11 - y_wu
	
	# Total potential energy
	Pi = psi1*dx
	
	# Compute first variation of Pi (directional derivative about u in the direction of v)
	R = derivative(Pi, u, v) 
	
	# Compute Jacobian of R
	Jac = derivative(R, u, du) 
	
	#To use later for memory allocation for these tensors
	A=PETScMatrix()
	b=PETScVector()
	
	#Balance of configurational forces PDE
	Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
	Wv2=conditional(le(z, 0.05), 1, 0)*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
	R_z = -Y_wu*y*dx + (1/(1+h/(pi*eps)))*Gc/pi*(y*(-2*z)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) + derivative(Wv2,z,y)
	
	# Compute Jacobian of R_z
	Jac_z = derivative(R_z, z, dz)

# ---------------------------------------------------------------------------------------------------------------------------------------------
# 4. Star Convex Model 2023		Model on Enegy Decomposition under Multi Axial Stress States  # Dr. Lorenzis
# ---------------------------------------------------------------------------------------------------------------------------------------------
elif phase_model == 4:
	eps = lch
	pen=1000*Gc/eps
	eta=1e-7
	gamma = 0; # (3*(sts/scs)**2-2*(nu+1))/(2*nu-1)
	
	def energy_pos(v):
		strain = epsilon(v)
		strain_dev=epsilon_dev(v)
		trstrain=tr(strain)
		tr_eps_plus = (1./2.)*(trstrain + abs(trstrain))
		tr_eps_neg = (1./2.)*(trstrain - abs(trstrain))
		#		energy_density = 0.5*kappa*(tr_eps_plus)**2 + mu*(inner(strain_dev,strain_dev)+(nu/(1-nu)*tr(strain))**2)-gamma*0.5*kappa*(tr_eps_neg)**2
		energy_density = 0.5*kappa*(tr_eps_plus)**2 + mu*inner(strain_dev,strain_dev)-gamma*0.5*kappa*(tr_eps_neg)**2
		return energy_density
	
	def energy_neg(v):
		strain = epsilon(v)
		trstrain=tr(strain)
		tr_eps_neg = (1./2.)*(trstrain - abs(trstrain))
		energy_density =(1+gamma)*0.5*kappa*(tr_eps_neg)**2
		return energy_density
	
	J2 = sigmavm(sigma(u),u)**2
	triaxiality= tr(sigma(u))/(sqrt(3*J2))
	# Stored strain energy density (compressible L-P model)
	psi1 =(z**2+eta)*(energy_pos(u))+energy_neg(u)	
	psi11=energy_pos(u) #*(1+sqrt((triaxiality+0)**2)/(triaxiality+0)/2)
	stress=(z**2+eta)*sigma(u)

	f1 = (scs-sts)/(sqrt(3)*(scs + sts))
	f0 = -(2*scs*sts)/(sqrt(3)*(scs + sts))
	FDP = (sigmavm(sigma(u),u)) + f1*(tr(sigma(u))) + f0
	FDP2 = 2*psi11 - 3*Gc/8/eps
	indicator=conditional(gt(FDP, 0), 1, 0)
	indicator2=conditional(gt(FDP2, 0), 1, 0)
	
	# Total potential energy
	Pi = psi1*dx
	# Compute first variation of Pi (directional derivative about u in the direction of v)
	R = derivative(Pi, u, v) 
	
	# Compute Jacobian of R
	Jac = derivative(R, u, du)
	
	
	
	#To use later for memory allocation for these tensors
	A=PETScMatrix()
	b=PETScVector()

	#Balance of configurational forces PDE
	Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
	Wv2=conditional(le(z, 0.05), 1, 0)*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
	R_z =y*2*z*(psi11)*dx + 3*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) + derivative(Wv2,z,y)
	
	# Compute Jacobian of R_z
	Jac_z = derivative(R_z, z, dz)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 5. Classical AT1 model
# ---------------------------------------------------------------------------------------------------------------------------------------------	
	
elif phase_model == 5: #classical model without an energy split
	eps = lch
	pen=1000*Gc/eps
	eta=1e-7
	
	
	J2 = sigmavm(sigma(u),u)**2
	triaxiality= tr(sigma(u))/(sqrt(3*J2))
	# Stored strain energy density (compressible L-P model)
	psi1 =(z**2+eta)*(energy(u))	
	psi11=energy(u) #*(1+sqrt((triaxiality+0)**2)/(triaxiality+0)/2)
	stress=(z**2+eta)*sigma(u)

	f1 = (scs-sts)/(sqrt(3)*(scs + sts))
	f0 = -(2*scs*sts)/(sqrt(3)*(scs + sts))
	FDP = (sigmavm(sigma(u),u)) + f1*(tr(sigma(u))) + f0
	FDP2 = 2*psi11 - 3*Gc/8/eps
	indicator=conditional(gt(FDP, 0), 1, 0)
	indicator2=conditional(gt(FDP2, 0), 1, 0)
	
	# Total potential energy
	Pi = psi1*dx
	# Compute first variation of Pi (directional derivative about u in the direction of v)
	R = derivative(Pi, u, v) 
	
	# Compute Jacobian of R
	Jac = derivative(R, u, du)
	
	
	
	#To use later for memory allocation for these tensors
	A=PETScMatrix()
	b=PETScVector()

	#Balance of configurational forces PDE
	Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
	Wv2=conditional(le(z, 0.05), 1, 0)*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
	R_z =y*2*z*(psi11)*dx + 3*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) + derivative(Wv2,z,y)
	
	# Compute Jacobian of R_z
	Jac_z = derivative(R_z, z, dz)
	
	
# ---------------------------------------------------------------------------------------------------------------------------------------------
# 6. Spectral split model
# ---------------------------------------------------------------------------------------------------------------------------------------------	
	
elif phase_model == 6: 

	eps = lch
	pen=1000*Gc/eps
	eta=1e-7
	strain = epsilon(u)
	
	def spec_split_energy_pos(trstrain, eigenvals):
		tr_eps_plus = 0.5 * (trstrain + abs(trstrain))
		_sum = sum((0.5 * (eig + abs(eig)))**2 for eig in eigenvals)
		return 0.5 * lmbda * tr_eps_plus**2 + mu *_sum

	def spec_split_energy_neg(trstrain, eigenvals):
		tr_eps_neg = 0.5 * (trstrain - abs(trstrain))
		_sum = sum((0.5 * (eig - abs(eig)))**2 for eig in eigenvals)
		return 0.5 * lmbda * tr_eps_neg**2 + mu * _sum
	
	
	eigenvals = list(compute_eigenvalues(strain, 3))
	
	def energy_pos(v):
		strain=epsilon(v)
		return spec_split_energy_pos(tr(strain), eigenvals)

	def energy_neg(v):
		strain=epsilon(v)
		return spec_split_energy_neg(tr(strain), eigenvals)
	
	# J2 = sigmavm(sigma(u),u)**2
	# triaxiality= tr(sigma(u))/(sqrt(3*J2))
	# Stored strain energy density (compressible L-P model)
	psi1 = (z ** 2 + eta) * energy_pos(u) + energy_neg(u)
	psi11 = energy_pos(u)
	stress = (z ** 2 + eta) * sigma(u)

	f1 = (scs-sts)/(sqrt(3)*(scs + sts))
	f0 = -(2*scs*sts)/(sqrt(3)*(scs + sts))
	FDP = (sigmavm(sigma(u),u)) + f1*(tr(sigma(u))) + f0
	FDP2 = 2*psi11 - 3*Gc/8/eps
	indicator=conditional(gt(FDP, 0), 1, 0)
	indicator2=conditional(gt(FDP2, 0), 1, 0)
	
	# Total potential energy
	Pi = psi1*dx
	# Compute first variation of Pi (directional derivative about u in the direction of v)
	R = derivative(Pi, u, v) 
	
	# Compute Jacobian of R
	Jac = derivative(R, u, du)
	
	
	
	#To use later for memory allocation for these tensors
	A=PETScMatrix()
	b=PETScVector()

	#Balance of configurational forces PDE
	Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
	Wv2=conditional(le(z, 0.05), 1, 0)*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
	R_z =y*2*z*(psi11)*dx + 3*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) + derivative(Wv2,z,y)
	
	# Compute Jacobian of R_z
	Jac_z = derivative(R_z, z, dz)


#To use later for memory allocation for these tensors
A=PETScMatrix()
b=PETScVector()

# Define the solver parameters
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "cg",   
                                          "preconditioner": "petsc_amg",						  
                                          "maximum_iterations": 10,
										  "absolute_tolerance":1e-8,
                                          "report": True,
                                          "error_on_nonconvergence": False}}			


solver_u = KrylovSolver('cg', 'petsc_amg')
# max_iterations=450
# solver_u.parameters["maximum_iterations"] = max_iterations
solver_u.parameters["error_on_nonconvergence"] = False

#time-stepping parameters
T=1
Totalsteps=200
startstepsize=1/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
rtol=1e-8
tau=0
Wext=0
while t-stepsize < T:

	if comm_rank==0:
		print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)
		sys.stdout.flush()
	
	r.t=t; ry.t=t; r0.t=t; r0.tau=tau;
	
	stag_iter=1
	rnorm_stag=1
	while stag_iter<50 and rnorm_stag > 1e-7:
		start_time=time.time()
		##############################################################
		#First PDE
		##############################################################		
		# Problem_u = NonlinearVariationalProblem(R, u, bcs, J=Jac)
		# solver_u  = NonlinearVariationalSolver(Problem_u)
		# solver_u.parameters.update(snes_solver_parameters)
		# (iter, converged) = solver_u.solve()
		
		terminate=0
		##Newton iterations##################################
		nIter = 0
		rnorm = 10000.0
		rnorm_prev=10000.0
			
		while nIter < 10: 	
			nIter += 1
			
			# if nIter>20 and terminate2==0:
				# terminate=1
				# break
			
			
			if nIter==1 and stag_iter==1:
				A, b = assemble_system(Jac, -R, bcs_du0)
			else:
				A, b = assemble_system(Jac, -R, bcs_du)
			
			rnorm=b.norm('l2')
				
			
			if comm_rank==0:	
				print('Iteration number= %d' %nIter,  'Residual= %e' %rnorm)
				
			if rnorm < rtol:             #residual check
				break
			
			# if rnorm > rnorm_prev*8 or np.isnan(rnorm) is True:          
				# terminate=1
				# break
			
			# if nIter==20 and rnorm>1e0:
				# terminate=1
				# break
			
			
			# rnorm_prev=rnorm

			
			# start_time=time.time()
			
			converged = solver_u.solve(A, u_inc.vector(), b);
			# if comm_rank==0:
				# print("--- %s seconds ---" % (time.time() - start_time))
				# print(converged)
			
			u.vector().axpy(1, u_inc.vector())    #very fast
			
		##############################################################
		#Second PDE
		##############################################################
		Problem_z = NonlinearVariationalProblem(R_z, z, bcs_z, J=Jac_z)
		solver_z  = NonlinearVariationalSolver(Problem_z)
		solver_z.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_z.solve()
			
		min_z = z.vector().min();
		zmin = MPI.min(comm, min_z)
		if comm_rank==0:
			print(zmin)
		
		if comm_rank==0:
			print("--- %s seconds ---" % (time.time() - start_time))
			sys.stdout.flush()

	  
		###############################################################
		#Residual check for stag loop
		###############################################################
		b=assemble(-R, tensor=b)
		fint=b.copy() #assign(fint,b) 
		for bc in bcs_du:
			bc.apply(b)
		rnorm_stag=b.norm('l2')	
		stag_iter+=1  

		
	######################################################################
	#Post-Processing
	######################################################################
	Fx=MPI.sum(comm,sum(fint[y_dofs_top]))
	Wext = Wext - Fx*disp*stepsize # assemble((dot(dot(stress,n),(u-u_prev)))*ds(3))
	Strain_energy = assemble(psi1 * dx)
	dissipated_energy=Wext-Strain_energy
	crack_area=dissipated_energy/Gc
	surf_area = assemble(3/8*((1-z)/eps + eps*inner(grad(z),grad(z)))*dx)
	
	assign(u_prev,u)
	assign(z_prev,z)
	tau+=stepsize
	
	
    ####Calculate Reaction

	if comm_rank==0:
		printdata=[t, t*disp, Fx, Strain_energy, Wext, dissipated_energy, crack_area, surf_area ]
		with open('Echelon_' + filemodel + '.csv', 'a', newline='') as csvfile: 
			csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			csvwriter.writerow(printdata)
	
	if step % 5==0:
		Yv=FunctionSpace(mesh, "DG", 0)
		Jprint=Function(Yv)
		local_project(J, Yv, Jprint)
		# indprint1=project(indicator,Yv)
		# indprint2=project(indicator2,Yv)

		file_results = XDMFFile( "paraview/Echelon_" + filemodel + "_" + str(step) + ".xdmf" )
		file_results.parameters["flush_output"] = True
		file_results.parameters["functions_share_mesh"] = True

		u.rename("u", "displacement field")
		z.rename("z", "phase field")
		file_results.write(u,t)
		file_results.write(z,t)
		
		Jprint.rename("J", "Det(F)")
		# file_results.write(Jprint,t)
	
	if step % 5==0:
		fFile = HDF5File(comm,"solz_" + filemodel + "_" + str(step) +".h5","w")
		# fFile.write(u,"/u")
		fFile.write(z,"/z")
		fFile.close()
		

		# indprint1.rename("flag1", "surface flag1")
		# indprint2.rename("flag2", "surface flag2")
		# file_results.write(indprint1,t)
		# file_results.write(indprint2,t)

	#time stepping
	step+=1
	t+=stepsize




