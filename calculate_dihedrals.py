import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import calc_dihedral
import numpy as np
import pickle
import os, traceback
from tqdm import tqdm

PDB_DIR='/rds/project/rds-a1NGKrlJtrw/dyna_mol/v2020-other-PL/'
DATA_DIR='/rds/project/rds-a1NGKrlJtrw/dyna_mol/data_PDBBind_1024/'
DIHEDRAL_DIR='/home/fmvb2/dihedral_folder/'

chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = {
    "ALA": [0.0, 0.0, 0.0, 0.0],  # ALA
    "ARG":[1.0, 1.0, 1.0, 1.0],  # ARG
    "ASN":[1.0, 1.0, 0.0, 0.0],  # ASN
    "ASP":[1.0, 1.0, 0.0, 0.0],  # ASP
    "CYS":[1.0, 0.0, 0.0, 0.0],  # CYS
    "GLN":[1.0, 1.0, 1.0, 0.0],  # GLN
    "GLU":[1.0, 1.0, 1.0, 0.0],  # GLU
    "GLY":[0.0, 0.0, 0.0, 0.0],  # GLY
    "HIS":[1.0, 1.0, 0.0, 0.0],  # HIS
    "ILE":[1.0, 1.0, 0.0, 0.0],  # ILE
    "LEU":[1.0, 1.0, 0.0, 0.0],  # LEU
    "LYS":[1.0, 1.0, 1.0, 1.0],  # LYS
    "MET":[1.0, 1.0, 1.0, 0.0],  # MET
    "PHE":[1.0, 1.0, 0.0, 0.0],  # PHE
    "PRO":[1.0, 1.0, 0.0, 0.0],  # PRO
    "SER":[1.0, 0.0, 0.0, 0.0],  # SER
    "THR":[1.0, 0.0, 0.0, 0.0],  # THR
    "TRP":[1.0, 1.0, 0.0, 0.0],  # TRP
    "TYR":[1.0, 1.0, 0.0, 0.0],  # TYR
    "VAL":[1.0, 0.0, 0.0, 0.0],  # VAL
}

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.

aas=["ALA",
"ARG",
"ASN",
"ASP",
"CYS",
"GLU",
"GLN",
"GLY",
"HIS",
"HSD",
"HSE",
"ILE",
"LEU",
"LYS",
"MET",
"PHE",
"PRO",
"SER",
"THR",
"TRP",
"TYR",
"VAL"]

def get_filelist():
    filelist = []
    for subdir, dirs, files in os.walk(PDB_DIR):
        for file in files:
            if 'protein.pdb' in file:
                filelist.append(file)

    return filelist

wf=open('error_pdb_dihedrals.txt','w')

def vectors_to_dihedral(vectors):
    # Bio calculates dihedral in radians i [-π, π]
    single_dihedral = calc_dihedral(vectors[0], vectors[1], vectors[2], vectors[3])
    # Project the dihedral onto the unit circle
    dihedral = [np.sin(single_dihedral), np.cos(single_dihedral)]
    return dihedral

filelist = get_filelist()
error_files = []

for file in tqdm(filelist):
    protein_id = file.replace('_protein.pdb', '')
    print(f'id: f{protein_id}')
    filepath = os.path.join(PDB_DIR, protein_id, file)
    parser=PDBParser(QUIET=True)
    structure=parser.get_structure(protein_id, filepath)
    
    residues= structure.get_residues()
    protein_res_torsions=[]
    aa_list=[]
    
    for resi in residues:
        resi_name = resi.get_resname() 
        if resi_name not in aas: continue
        torsion_angles = np.zeros([4, 2])
        try:
            atoms_in_dihedral = chi_angles_atoms[resi_name] # Shape [n_torsions, 4]
            for idx, chi in enumerate(atoms_in_dihedral):
                dihedral_atom_coords = [resi[atom].get_vector() for atom in chi] # Shape [4, 3]
                dihedral = vectors_to_dihedral(dihedral_atom_coords)
                torsion_angles[idx, :] = dihedral
            
            protein_res_torsions.append(torsion_angles)
            aa_list.append(resi_name)
            
            #if not torsion_angles:
                #print(f"empty torsion: {torsion_angles}, res = {resi_name}, protein = {protein_id}")

        except Exception as e:

            #traceback.print_exc()
            wf.write(f'torsion error: {protein_id}, {resi_name}' + '\n')
            protein_res_torsions.append([[0,0], [0,0], [0,0], [0,0]])
            aa_list.append(resi_name)
        try:
            assert len(protein_res_torsions) == len(residues)
        except:
            if file not in error_files:
                wf.write(file+'\n')
                error_files.append(file)

            continue
        
       
    dihedral_angles={'dihedrals': protein_res_torsions, 'aa_list': aa_list}
    with open(DIHEDRAL_DIR + protein_id + '.pickle', 'wb') as wbf:
        pickle.dump(dihedral_angles, file=wbf)
