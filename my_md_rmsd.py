#!/usr/bin/env python
import os

# Default: one thread outside xTB calculations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import subprocess
import numpy as np
import time
from xtb.interface import Calculator, Param
from xtb.libxtb import VERBOSITY_MUTED
from threadpoolctl import ThreadpoolController


# Create after importing NumPy and xTB.
thread_controller = ThreadpoolController()
XTB_CORES = 4



#from spyrmsd import rmsd


kB = 1.380649e-23          # J/K
u = 1.66053906892e-27   # kg per atomic mass unit
BOHR_TO_ANGSTROM = 0.529177210903
hartree_J = 4.359744722206e-18
working_dir = Path.cwd()

atomic_number = {'H': 1, 'Li': 3, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
#atomic_weight = {'H': 1.008, 'Li': 6.941, 'B': 10.811, 'C': 12.011,
atomic_weight = {'H': 4.000, 'Li': 6.941, 'B': 10.811, 'C': 12.011,
                 'N': 14.006, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
                 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si':  28.086,
                 'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
                 'K': 39.098, 'Ca': 40.078, 'Ti': 47.867, 'Fe': 55.845,
                 'Zn': 65.38, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
                 'Mo': 95.96, 'Ru': 101.07, 'Sn': 118.710, 'Te': 127.60,
                 'I': 126.904, 'Xe': 131.293}


def read_xyz(filename):
    atoms = np.empty(0,dtype=object)
    numbers = np.empty(0,dtype=np.int32)
    masses = np.empty(0,dtype=np.float32)
    coords = np.empty((0,3),dtype=np.float32)

    f = open(filename)
    natoms = int(f.readline())
    title = f.readline()
    for line in f:
        atom,x,y,z = line.split()
        atoms = np.append(atoms,np.array(atom))
        numbers = np.append(numbers,np.array(atomic_number[atom]))
        masses = np.append(masses,np.array(atomic_weight[atom]))
        coords = np.append(coords,np.array([[float(x),float(y),float(z)]]),axis=0)
    f.close()

    return atoms,numbers,masses,coords


def get_initial_velocities(natoms,masses,temp):
    vels = np.zeros((natoms,3),dtype=np.float32)
    for i in range(natoms):
        # Convert m/s to A/fs, include kB and u (atomic mass)
        vels[i] = np.random.normal(0.0,0.000911836751256*np.sqrt(temp/masses[i]),size=3)
    return vels   # potentially: remove center-of-mass velocity


def read_orca_gradients(natoms):
    gradients = np.zeros((natoms,3),dtype=np.float32)

    f = open("orca.engrad")
    for i in range(11):
        line = f.readline()
    for i in range(natoms):
        for j in range(3):
            line = f.readline()
            grad = float(line.rstrip())
            gradients[i,j] = grad
    f.close()

    return -gradients


def get_acceleration(forces,masses):
    accs = np.zeros((natoms,3),dtype=np.float32)
    for i in range(natoms):
        # Convert atomic units (force by mass) to A/fs**2
        accs[i] = 0.496147525704*(forces[i]/masses[i])
    return accs


def write_xyz(filename,atoms,coords):
    f = open(filename + ".xyz","w")
    f.write("{}\n\n".format(len(coords)))
    for j in range(len(coords)):
        atom = atoms[j]
        x,y,z = coords[j]
        f.write("{} {} {} {}\n".format(atom,x,y,z))
    f.close()


def append_xyz(filename,atoms,coords):
    f = open(filename + ".xyz","a")
    f.write("{}\n\n".format(len(coords)))
    for j in range(len(coords)):
        atom = atoms[j]
        x,y,z = coords[j]
        f.write("{} {} {} {}\n".format(atom,x,y,z))
    f.close()


def write_initial_orca_input(charge,multi,pals,functional,atoms,coords):
    f = open("orca.inp","w")
    f.write("! " + functional + " TIGHTSCF EnGrad MiniPrint NoPop \n%maxcore 1000\n%pal\nnprocs "+ pals +"\nend\n%scf\nFracOcc true\nSmearTemp 6000\nend\n")
    f.write("%output\n  PrintLevel Mini\n  Print[P_OrbEn] 0\n  Print[P_Cartesian] 0\n  Print[P_InputFile] 0\nend\n")
    f.write("%elprop\n  Dipole false\n  Quadrupole false\nend\n")
    f.write("* xyz {} {}\n".format(charge,multi))
    for j in range(len(atoms)):
        atom = atoms[j]
        x,y,z = coords[j]
        f.write("{} {} {} {}\n".format(atom,x,y,z))
    f.write("*\n")
    f.close()


def write_orca_input(charge,multi,pals,functional,atoms,coords):
    f = open("orca.inp","w")
    f.write("! " + functional + " TIGHTSCF EnGrad MORead MiniPrint NoPop \n%maxcore 1000\n%moinp \"orca_pre.gbw\"\n%pal\nnprocs "+ pals +"\nend\n%scf\nFracOcc true\nSmearTemp 6000\nend\n")
    f.write("%output\n  PrintLevel Mini\n  Print[P_OrbEn] 0\n  Print[P_Cartesian] 0\n  Print[P_InputFile] 0\nend\n")
    f.write("%elprop\n  Dipole false\n  Quadrupole false\nend\n")
    f.write("* xyz {} {}\n".format(charge,multi))
    for j in range(len(atoms)):
        atom = atoms[j]
        x,y,z = coords[j]
        f.write("{} {} {} {}\n".format(atom,x,y,z))
    f.write("*\n")
    f.close()


def calc_xtb_gradients(coords,calc, step, previous_result=None):
    
    """Coordinates in angstrom; gradient returned in Hartree/bohr."""
    calc.update(
        np.asarray(coords, dtype=np.float64) / BOHR_TO_ANGSTROM
    )
    with thread_controller.limit(limits=XTB_CORES):
        result = calc.singlepoint(previous_result)
    energy = result.get_energy()       # Hartree
    forces = -result.get_gradient()    # Hartree/bohr; shape (natoms, 3)

    return forces, result

def calc_temperature(natoms,masses,vels):
    s = 0.0
    for i in range(natoms):
        s += masses[i]*np.sum(vels[i]*vels[i])
    # Convert A/fs to m/s, include u (atomic mass) and kB
    return 0.000120272355169*1.0e+10*s/(3.0*natoms)


def calc_langevin_friction(natoms,gamma,vels):
    afrxn = np.zeros((natoms,3),dtype=np.float32)
    for i in range(natoms):
        # This expression has already been divided by mass, thus being an acceleration
        afrxn[i] = -gamma*vels[i]
    return afrxn


def calc_langevin_random(natoms,gamma,temp,masses,dt):
    arand = np.zeros((natoms,3),dtype=np.float32)
    for i in range(natoms):
        # This expression has already been divided by mass, thus being an acceleration
        # Convert m/s to A/fs (gamma and dt already in fs)
        arand[i] = np.random.normal(0.0,1.0e-5*np.sqrt(2.0*kB*temp*gamma/(masses[i]*u*dt)),size=3)
    return arand

def berendsen_rescale(vels, masses, temp, dt, tau_T):
    """Rescale velocities toward target temperature temp."""
    if tau_T <= 0 or dt <= 0:
        raise ValueError("dt and tau_T must be positive.")
    if temp <= 0:
        raise ValueError("Target temperature must be positive.")

    current_temp = calc_temperature(len(vels), masses, vels)

    if current_temp <= 0:
        raise ValueError("Cannot rescale velocities at zero temperature.")

    scale_squared = 1.0 + (dt / tau_T) * (
        temp / current_temp - 1.0
    )

    if scale_squared <= 0:
        raise ValueError("Invalid scaling: increase tau_T or decrease dt.")

    return vels * np.sqrt(scale_squared)


def calc_center_of_mass(atoms,coords):
    x = np.zeros(3,dtype=np.float32)
    mass = 0.0
    for i in range(len(coords)):
        m = atomic_weight[atoms[i]]
        x += m*coords[i]
        mass += m
    return x/mass


def calc_center_of_interaction(coords):
    x = np.zeros(3,dtype=np.float32)
    norm = 0.0
    for i in range(len(coords)):
        x += coords[i]
        norm += 1.0
    return x/norm


def update_velocities(vels,accs,dt):
    return vels + 0.5*accs*dt


def update_positions(coords,vels,dt):
    return coords + vels*dt





def calc_confinement(natoms, coords, masses, rsphere, beta, temp, return_energy=False):
    """
    xTB-style spherical log-Fermi wall centered at the origin.

    Inputs:
        coords:  Cartesian coordinates in angstrom
        masses:  atomic masses in amu
        rsphere: cavity radius in angstrom
        beta:    wall steepness in angstrom^-1
        temp:    wall-strength parameter in kelvin

    Returns:
        Accelerations in angstrom/fs^2.
        If return_energy=True: (accelerations, wall_energy_Hartree).
    """
    coords = np.asarray(coords, dtype=np.float64)
    accs = np.zeros((natoms, 3), dtype=np.float64)
    energy_J = 0.0


    for i in range(natoms):
        r = np.linalg.norm(coords[i])
        x = beta * (r - rsphere)

        # V_i = kB*T*log(1 + exp(x)), evaluated without overflow
        energy_J += kB * temp * np.logaddexp(0.0, x)

        if r == 0.0:
            continue

        # exp(x)/(1 + exp(x)), evaluated without overflow
        sigmoid = np.exp(-np.logaddexp(0.0, -x))

        # Negative energy gradient / mass.
        # 1e-10 converts J/angstrom/amu to angstrom/fs^2,
        # with the amu-to-kg conversion supplied by u.
        accs[i] = (
            -1.0e-10 * kB * temp * beta
            * sigmoid / (masses[i] * u)
            * coords[i] / r
        )

    if return_energy:
        return accs, energy_J / hartree_J

    return accs

def calc_metadynamics_rmsd(natoms,numbers,masses,coords,refcoords,refcount,kpush,alpha,exclude_atom,refcoords_perm,kpush_perm,alpha_perm):
    if len(exclude_atom) > 0:
        natoms -= len(exclude_atom)
        for ex in exclude_atom:
            masses = np.delete(masses,ex)
            coords = np.delete(coords,ex,axis=0)
            refcoords = np.delete(refcoords,ex,axis=1)
            refcoords_perm = np.delete(refcoords_perm,ex,axis=1)

    fbym = np.zeros((natoms,3),dtype=np.float32)
    acc = np.zeros((natoms, 3),dtype=np.float32)

    # Remove translational degrees of freedom for current coordinates
    x = coords[:] - calc_center_of_interaction(coords[:])

    for i in range(len(refcoords)):
        # Scaling factor
        fac = 2.0 / ( 1.0 + np.exp(-0.03*refcount[i]) ) - 1.0

        # Remove translational degrees of freedom for reference coordinates
        y = refcoords[i][:] - calc_center_of_interaction(refcoords[i][:])
        #val2 = rmsd.rmsd(x,y,numbers,numbers,True,True,1e-9)

        # Correlation matrix
        R = np.zeros((3,3),dtype=np.float32)
        for j in range(natoms):
            R += np.outer(x[j],y[j])

        # Quaternion matrix
        F = np.zeros((4,4),dtype=np.float32)
        # Diagonal elements
        F[0,0] = R[0,0] + R[1,1] + R[2,2]
        F[1,1] = R[0,0] - R[1,1] - R[2,2]
        F[2,2] = -R[0,0] + R[1,1] - R[2,2]
        F[3,3] = -R[0,0] - R[1,1] + R[2,2]
        # Off-diagonal elements
        F[0,1] = F[1,0] = R[1,2] - R[2,1]
        F[0,2] = F[2,0] = R[2,0] - R[0,2]
        F[0,3] = F[3,0] = R[0,1] - R[1,0]
        F[1,2] = F[2,1] = R[0,1] + R[1,0]
        F[1,3] = F[3,1] = R[0,2] + R[2,0]
        F[2,3] = F[3,2] = R[1,2] + R[2,1]
        # Determine largest eigenvalue and corresponding eigenvector for rotation matrix
        eigenvalues,eigenvectors = np.linalg.eig(F)
        lmax = np.max(eigenvalues)
        imax = np.argmax(eigenvalues)
        RMSD = np.sqrt( ( np.sum(x*x) + np.sum(y*y) - 2.0*lmax )/ natoms )
        q = eigenvectors[:,imax]

        # Calculate rotation matrix
        U = np.zeros((3,3),dtype=np.float32)
        # Diagonal elements
        U[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        U[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
        U[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        # Off-diagonal elements
        U[0,1] = 2.0*( q[1]*q[2] - q[0]*q[3] )
        U[1,0] = 2.0*( q[1]*q[2] + q[0]*q[3] )
        U[0,2] = 2.0*( q[1]*q[3] + q[0]*q[2] )
        U[2,0] = 2.0*( q[1]*q[3] - q[0]*q[2] )
        U[1,2] = 2.0*( q[2]*q[3] - q[0]*q[1] )
        U[2,1] = 2.0*( q[2]*q[3] + q[0]*q[1] )
        # for j in range(natoms):
        #     fbym[j] += fac * ( x[j] - (U.T).dot(y[j]) ) / masses[j]  # Normalization is added below and RMSD value cancels

        delta = x - y @ U

        rmsd2_bohr = np.sum(delta * delta) / (
            natoms * BOHR_TO_ANGSTROM**2
        )

        gaussian = np.exp(-alpha * rmsd2_bohr)
        fbym += fac * gaussian * delta / masses[:, None]
    acc = (0.496147525704 * 2.0 * alpha * kpush / (natoms * BOHR_TO_ANGSTROM)) * fbym   #   0.496147525704 = Conversion from Eh/(bohr * amu) to A/fs**2
    fbym = np.zeros((natoms,3),dtype=np.float32)
    for i in range(len(refcoords_perm)):
        # No Scaling factor for permanent refcoords
        #fac = 2.0 / ( 1.0 + np.exp(-0.03*refcount[i]) ) - 1.0

        # Remove translational degrees of freedom for reference coordinates
        y = refcoords_perm[i][:] - calc_center_of_interaction(refcoords_perm[i][:])
        #val2 = rmsd.rmsd(x,y,numbers,numbers,True,True,1e-9)

        # Correlation matrix
        R = np.zeros((3,3),dtype=np.float32)
        for j in range(natoms):
            R += np.outer(x[j],y[j])

        # Quaternion matrix
        F = np.zeros((4,4),dtype=np.float32)
        # Diagonal elements
        F[0,0] = R[0,0] + R[1,1] + R[2,2]
        F[1,1] = R[0,0] - R[1,1] - R[2,2]
        F[2,2] = -R[0,0] + R[1,1] - R[2,2]
        F[3,3] = -R[0,0] - R[1,1] + R[2,2]
        # Off-diagonal elements
        F[0,1] = F[1,0] = R[1,2] - R[2,1]
        F[0,2] = F[2,0] = R[2,0] - R[0,2]
        F[0,3] = F[3,0] = R[0,1] - R[1,0]
        F[1,2] = F[2,1] = R[0,1] + R[1,0]
        F[1,3] = F[3,1] = R[0,2] + R[2,0]
        F[2,3] = F[3,2] = R[1,2] + R[2,1]
        # Determine largest eigenvalue and corresponding eigenvector for rotation matrix
        eigenvalues,eigenvectors = np.linalg.eig(F)
        lmax = np.max(eigenvalues)
        imax = np.argmax(eigenvalues)
        RMSD = np.sqrt( ( np.sum(x*x) + np.sum(y*y) - 2.0*lmax )/ natoms )
        q = eigenvectors[:,imax]

        # Calculate rotation matrix
        U = np.zeros((3,3),dtype=np.float32)
        # Diagonal elements
        U[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        U[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
        U[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        # Off-diagonal elements
        U[0,1] = 2.0*( q[1]*q[2] - q[0]*q[3] )
        U[1,0] = 2.0*( q[1]*q[2] + q[0]*q[3] )
        U[0,2] = 2.0*( q[1]*q[3] + q[0]*q[2] )
        U[2,0] = 2.0*( q[1]*q[3] - q[0]*q[2] )
        U[1,2] = 2.0*( q[2]*q[3] - q[0]*q[1] )
        U[2,1] = 2.0*( q[2]*q[3] + q[0]*q[1] )
        # for j in range(natoms):
        #     fbym[j] += fac * ( x[j] - (U.T).dot(y[j]) ) / masses[j]  # Normalization is added below and RMSD value cancels

        delta = x - y @ U

        rmsd2_bohr = np.sum(delta * delta) / (
            natoms * BOHR_TO_ANGSTROM**2
        )

        gaussian = np.exp(-alpha_perm * rmsd2_bohr)
        fbym += gaussian * delta / masses[:, None]
    acc += (0.496147525704 * 2.0 * alpha_perm * kpush_perm / (natoms * BOHR_TO_ANGSTROM)) * fbym
  
    if len(exclude_atom) > 0:
                for ex in exclude_atom:
                    acc = np.insert(acc, ex,np.array([[float(0),float(0),float(0)]]),axis=0)
    return acc


def read_input_file(input_file):
    coord_file = ""
    ref_coord_files = []
    kpush = 0.0
    alpha = 0.0
    kpush_perm = 0.0
    alpha_perm = 0.0
    with open(input_file) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            match key:
                case "coord":
                    coord_file = value.strip()
                case "ref_coord":
                    ref_coord_files = value.split()
                case "kpush":
                    kpush = float(value.strip())
                case "alpha":
                    alpha = float(value.strip())
                case "kpush_perm":
                    kpush_perm = float(value.strip())
                case "alpha_perm":
                    alpha_perm = float(value.strip())
                case _:
                    raise ValueError(f"Unknown input keyword: {key}")


    return (coord_file, ref_coord_files, kpush, alpha, kpush_perm, alpha_perm)
            

if __name__ == "__main__":

    ### Parser to read input file    
    parser = argparse.ArgumentParser(description="Run an MD simulation.")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input file.",
    )
    args = parser.parse_args()
    coord_file, ref_coord_files, kpush, alpha, kpush_perm, alpha_perm=read_input_file(args.input)
    print(coord_file, ref_coord_files, kpush, alpha, kpush_perm, alpha_perm)


    thermostat = "B"        # B for Berendsen and L for Langevin
    
    charge = 0                  # Charge
    multi = 2                   # Multiplicity
    ntime = 4000               # Number of time steps
    dump = 20                   # Print trajectory xyz after nth step
    dt = 0.5                    # Time step in fs

    temp = 300.0                # Temperature in K
    tau_T = 100                 # Berendsen Thermostat

    confining_temp = 1000       # Confining Temp in K
    gamma = 0.1                 # Friction coefficient in 1/fs
    rsphere = 2.0               # radius of confining sphere in A
    beta = 6.0 / 0.529177       # steepness of confining potential, converted form Bohrs (xtb) to A

    tau = 2000                  # Add Metadynamics reference structure each n-th step
    save = 20                   # Maximum images for Metadynamics
    #kpush = 0.1                 # Height of Gaussian in Hartree, this is not converted to SI units for reasons of convenience
    #alpha = 0.6                 # Width of Gaussian in Bohr^{-2}, this is not converted to A for reasons of convenience
    exclude_atom = [10]         # list of atom numbers excluded in RMSD potential (0 for the first atom)
    run_equi = True
    ntime_equi = 500
    #refcoords_xyz = ["fb_0.xyz"]
    #kpush_perm = 0.5
    #alpha_perm = 0.4

    grad_calc = "XTB"               # gradient calculator ("ORCA" or "XTB")
    functional = "r2SCAN-3c"        # functional if ORCA is used
    pals = "16"
    orca_path = "/usr/qc/ORCA6/orca_6_1_1_linux_x86-64_shared_openmpi418/"
    #orca_path = "/Applic.HPC/Easybuild/zen4/2023b/software/ORCA/6.0.1-gompi-2023b-avx2/bin/"
    
    start_program = time.perf_counter()
    atoms,numbers,masses,coords = read_xyz(coord_file)
    natoms = len(atoms)
    refcoords_perm = np.empty((0,natoms,3),dtype=np.float32)
    for xyz in ref_coord_files:
        _, _, _, coord = read_xyz(xyz)
        refcoords_perm = np.append(refcoords_perm, np.array([coord]),axis=0)

    print(refcoords_perm)
    coords -= calc_center_of_mass(atoms,coords)
    refcoords = np.empty((0,natoms,3),dtype=np.float32)
    refcount = np.empty(0,dtype=np.int32)
    vels = get_initial_velocities(natoms,masses,temp)
    elapsed_module = 0

    # Gradient Calculation
    if grad_calc == "XTB":
        calc = Calculator(
                    Param.GFN2xTB,
                    np.asarray(numbers, dtype=np.int32),
                    np.asarray(coords, dtype=np.float64) / BOHR_TO_ANGSTROM,
                    charge=float(charge),
                    uhf=int(multi - 1),  # Doublet → one unpaired electron
                )
        calc.set_verbosity(VERBOSITY_MUTED)
        calc.set_accuracy(1.0)
        calc.set_electronic_temperature(6000.0)
        forces, result = calc_xtb_gradients(coords,calc,0)
    if grad_calc == "ORCA":
        write_initial_orca_input(charge,multi,pals,functional,atoms,coords)
        subprocess.call(orca_path+"orca orca.inp > orca.out",shell=True)
        subprocess.call("cp -a orca.gbw orca_pre.gbw",shell=True)
        forces = read_orca_gradients(natoms)
   
    accs = get_acceleration(forces,masses)
    write_xyz("trajectory",atoms,coords)
    print('T_init = ',calc_temperature(natoms,masses,vels))
    start = time.perf_counter()

    # Equi Loop
    if run_equi == True:
        start_equi = time.perf_counter()
        for t in range(ntime_equi):
            # 1. Update half-step velocities
            vels = update_velocities(vels,accs,dt)
            # 2. Update positions
            coords = update_positions(coords,vels,dt)
            # 3. Calculate new forces & accelerations
            start_module = time.perf_counter()
            if grad_calc == "XTB":
                forces, result = calc_xtb_gradients(coords,calc,t,result)
            if grad_calc == "ORCA":
                write_orca_input(charge,multi,pals,functional,atoms,coords)
                subprocess.call(orca_path+"orca orca.inp > orca.out",shell=True)     
                subprocess.call("cp -a orca.gbw orca_pre.gbw",shell=True)
                forces = read_orca_gradients(natoms)  
            elapsed_module = time.perf_counter() - start_module
            # 4. Update half-step velocities
            if thermostat == "L":
                accs = get_acceleration(forces,masses) \
                    + calc_langevin_friction(natoms,gamma,vels) \
                    + calc_langevin_random(natoms,gamma,temp,masses,dt) \
                    + calc_confinement(natoms,coords,masses,rsphere,beta,confining_temp)
                vels = update_velocities(vels,accs,dt)
            else:
                accs = get_acceleration(forces,masses) \
                    + calc_confinement(natoms,coords,masses,rsphere,beta,confining_temp) 
                vels = update_velocities(vels,accs,dt)
                vels = berendsen_rescale(vels, masses, temp, dt, tau_T=100.0)      
            if t % dump == 0:
                append_xyz("trajectory",atoms,coords)
                elapsed = time.perf_counter() - start
                print(
                    f"t = {t * dt:10.2f} fs | "
                    f"T = {calc_temperature(natoms, masses, vels):8.2f} K | "
                    f"Total timings  = {elapsed:10.3f} s | "
                    f"Module timings  = {elapsed_module:10.3f} s | "
                    )  
                start = time.perf_counter()
                elapsed_module = 0
        append_xyz("trajectory",atoms,coords)
        write_xyz("equi",atoms,coords)
        elapsed_equi = time.perf_counter() - start_equi
        print(
            f"Equilibration run finished equilibrated structure stored in equi.xyz |" 
            f"Equilibration timings = {elapsed_equi:10.3f} s"
        )
    


    # Metadynamics Loop
    for t in range(ntime):
        if t % tau == 0:
            while len(refcoords) >= save:
                refcoords = np.delete(refcoords,0,axis=0)
                refcount = np.delete(refcount,0)
            refcoords = np.append(refcoords,np.array([coords]),axis=0)
            refcount = np.append(refcount,np.array(0))
            print(refcoords)
        # 1. Update half-step velocities
        vels = update_velocities(vels,accs,dt)
        # 2. Update positions
        coords = update_positions(coords,vels,dt)
        # 3. Calculate new forces & accelerations
        start_module = time.perf_counter()
        if grad_calc == "XTB":
            forces, result = calc_xtb_gradients(coords,calc,t,result)
        if grad_calc == "ORCA":
            write_orca_input(charge,multi,pals,functional,atoms,coords)
            subprocess.call(orca_path+"orca orca.inp > orca.out",shell=True)     
            subprocess.call("cp -a orca.gbw orca_pre.gbw",shell=True)
            forces = read_orca_gradients(natoms)
        elapsed_module += time.perf_counter() - start_module

        # 4. Update half-step velocities
        if thermostat == "L":
            accs = get_acceleration(forces,masses) \
                + calc_langevin_friction(natoms,gamma,vels) \
                + calc_langevin_random(natoms,gamma,temp,masses,dt) \
                + calc_confinement(natoms,coords,masses,rsphere,beta,confining_temp) \
                + calc_metadynamics_rmsd(natoms,numbers,masses,coords,refcoords,refcount,kpush,alpha,exclude_atom,refcoords_perm,kpush_perm,alpha_perm)
            vels = update_velocities(vels,accs,dt)
        else:
            accs = get_acceleration(forces,masses) \
                + calc_confinement(natoms,coords,masses,rsphere,beta,confining_temp) \
                + calc_metadynamics_rmsd(natoms,numbers,masses,coords,refcoords,refcount,kpush,alpha,exclude_atom,refcoords_perm,kpush_perm,alpha_perm)
            vels = update_velocities(vels,accs,dt)
            vels = berendsen_rescale(vels, masses, temp, dt, tau_T=100.0)

        # Print stuff
        if t % dump == 0:
            append_xyz("trajectory",atoms,coords)
            elapsed = time.perf_counter() - start
            print(
                f"t = {t * dt:10.2f} fs | "
                f"T = {calc_temperature(natoms, masses, vels):8.2f} K | "
                f"Total timings  = {elapsed:10.3f} s | "
                f"Module timings  = {elapsed_module:10.3f} s | "
                )
            start = time.perf_counter()
            elapsed_module = 0
        refcount += np.ones(len(refcount),dtype=np.int32)
    program_timing = time.perf_counter()-start_program
    hours, remainder = divmod(round(program_timing, 3), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f"Overall timing = {int(hours):02d} h "
        f"{int(minutes):02d} min {seconds:06.3f} s"
    )
            
        


