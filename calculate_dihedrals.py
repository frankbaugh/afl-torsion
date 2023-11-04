import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import calc_dihedral
import numpy as np
import pickle
import os
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
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

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
#print(filelist)
    return filelist

wf=open('error_pdb_dihedrals.txt','w')

def vectors_to_dihedral(vectors):
    # Bio calculates dihedral in radians i [-π, π]
    single_dihedral = calc_dihedral(vectors[0], vectors[1], vectors[2], vectors[3])
    # Project the dihedral onto the unit circle
    dihedral = [np.sin(single_dihedral), np.cos(single_dihedral)]
    return dihedral

filelist = get_filelist()
for file in tqdm(filelist):
    protein_id = file.replace('_protein.pdb', '')
    print(f'id: f{protein_id}')
    filepath = os.path.join(PDB_DIR, protein_id, file)
    print(filepath)
    parser=PDBParser()
    structure=parser.get_structure(protein_id, filepath)
    
    residues= structure.get_residues() # Is this the same as below?
    #for chain in list(structure[0].get_chains()):
    #    residues+=list(chain.get_residues()
    protein_res_torsions=[]
    protein_res_masks=[]
    
    for resi in residues:
        resi_name = resi.get_resname() 
        if resi_name not in aas: continue
        torsion_angles = []
        try:
            atoms_in_dihedral = chi_angles_atoms[resi_name] # Shape [n_torsions, 4]
            for chi in atoms_in_dihedral:
                dihedral_atom_coords = [resi[atom].get_vector() for atom in chi] # Shape [4, 3]
                assert dihedral_atom_coords
                dihedral = vectors_to_dihedral(dihedral_atom_coords)
                torsion_angles.append(dihedral)
            
            protein_res_torsions.append(torsion_angles)
            protein_res_masks.append(chi_angles_mask[resi_name])
            
            #if not torsion_angles:
                #print(f"empty torsion: {torsion_angles}, res = {resi_name}, protein = {protein_id}")

        except Exception as e:

            print(e)
            wf.write(f'torsion error: {protein_id}, {resi_name}' + '\n')
            protein_res_torsions.append([[0,0], [0,0], [0,0], [0,0], [0,0]])
            protein_res_masks.append([0,0,0,0,0])
        try:
            assert len(protein_res_torsions) == len(residues)
        except:
            wf.write(file+'\n')
            continue
        
        
    dihedral_angles={'dihedrals': protein_res_torsions, 'dihedral_masks': protein_res_masks}
    with open(DIHEDRAL_DIR + protein_id + '.pickle', 'wb') as wbf:
        pickle.dump(dihedral_angles, file=wbf)
