# %%
import numpy as np
import os
import subprocess
from rdkit import Chem
atomic_number = {'H': 1, 'Li': 3, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
r0 = {
    (1, 1): 0.74,  # H–H
    (1, 3): 1.60,  # H–Li
    (1, 6): 1.10,  # H–C
    (1, 8): 0.96,  # H–O
    (3, 3): 2.67,  # Li–Li
    (3, 6): 2.10,  # Li–C, approximate
    (3, 8): 2.00,  # Li–O coordination, approximate
    (6, 6): 1.54,  # C–C single bond
    (6, 8): 1.43,  # C–O single bond
    (8, 8): 1.48,  # O–O single bond
}

class Molecule():
    def __init__(self,natoms,atoms,atnums,coords):
        self.natoms = natoms
        self.atoms = atoms
        self.atnums = atnums
        self.coords = coords

def calc_distance(coord1,coord2):
    return np.linalg.norm(coord2-coord1)

def read_trajectory_file(traj_file="trajectory.xyz"):
    nlines = 0
    natoms = 0
    nframes = 0
    mol_list = []
    exclude_atom = [10]
    with open(traj_file) as f:
        natoms = int(f.readline())
        nlines = int(sum(1 for _ in f)+1)
        print(int(nlines)/(int(natoms)+2))
        nframes = int(int(nlines)/(int(natoms)+2))
        
        f.seek(0)
        for i in range(nframes):
            atoms = np.empty(0,dtype=object)
            atnums = np.empty(0,dtype=np.int32)
            coords = np.empty((0,3),dtype=np.float32)
            natoms = int(f.readline())
            title = f.readline()

            for a in range(natoms):
                if a in exclude_atom:
                    fill = f.readline()
                    continue
                atom, x,y,z = f.readline().split()
                coords = np.append(coords,np.array([[float(x),float(y),float(z)]]),axis=0)
                atoms = np.append(atoms, atom)
                atnums = np.append(atnums, atomic_number[atom])
            mol_list.append(Molecule(natoms-len(exclude_atom),atoms,atnums,coords))
    return mol_list

def calc_adjmatrix(mol, offset):
    natoms = mol.natoms
    atnums = mol.atnums
    coords = mol.coords
    adjmatrix = np.zeros((natoms,natoms),dtype=np.float32)
    for i in range(natoms):
        for j in range(natoms):
            if i == j:
                continue
            pair = (tuple(sorted((mol.atnums[i],mol.atnums[i]))))
            if calc_distance(coords[i],coords[j]) < r0[pair] + offset:
                adjmatrix[i,j] = 1
            # adjmatrix[i,j] =  
    # with np.printoptions(precision=4, suppress=True, linewidth=200):
    #     print(adjmatrix)
    return adjmatrix

def adjacency_to_smiles(adjmatrix, symbols):
    adjmatrix = np.asarray(adjmatrix)
    natoms = len(symbols)

    if adjmatrix.shape != (natoms, natoms):
        raise ValueError("Expected one row and column per atom.")
    if not np.array_equal(adjmatrix, adjmatrix.T) or np.any(adjmatrix.diagonal()):
        raise ValueError("Expected a symmetric matrix with a zero diagonal.")

    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    mol = Chem.RWMol()
    for symbol in symbols:
        mol.AddAtom(Chem.Atom(symbol))

    # Read the upper triangle so each bond is added once.
    rows, cols = np.nonzero(np.triu(adjmatrix, k=1))
    for i, j in zip(rows, cols):
        order = adjmatrix[i, j]
        if order not in bond_types:
            raise ValueError(f"Unsupported bond order {order} at ({i}, {j}).")
        mol.AddBond(int(i), int(j), bond_types[order])

    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(Chem.RemoveHs(mol))
    
if __name__ == "__main__":
    ############## settings ##############
    offset = 0.1        
    nsameframes = 5    # number of same adjmatrix of consecutive frames to classify a molecule 
    ######################################
    mol_list = read_trajectory_file()
    adjmatrix_list = []
    natoms = mol_list[0].natoms
    adjmatrix_last = np.empty((natoms,natoms),dtype=np.float32)
    adjmatrix_test = np.empty((natoms,natoms),dtype=np.float32)
    #for i in range(len(mol_list)):
    count = 0
    for i in range(800):
        adjmatrix = calc_adjmatrix(mol_list[i], offset = 0.1)
        if i == 0:
            adjmatrix_last = calc_adjmatrix(mol_list[i], offset = 0.1)
            print(adjmatrix_last)
            continue

        if count > 0:
            if np.array_equal(adjmatrix_test,adjmatrix):
                if count >= nsameframes:
                    adjmatrix_list.append(calc_adjmatrix(mol_list[i], offset = 0.1))
                    adjmatrix_last = adjmatrix
                    print(i)
                    count = 0
                else:
                    count += 1
            else:
                count = 0
            continue
        if np.array_equal(adjmatrix,adjmatrix_last):
            # adjmatrix_list.append(calc_adjmatrix(mol_list[i], offset = 0.1))
            continue
        if not np.array_equal(adjmatrix,adjmatrix_last):
            adjmatrix_test = adjmatrix
            count = 1
    for adj in adjmatrix_list:
        print(adjacency_to_smiles(adj,mol_list[0].atoms))
            
            

# %%
