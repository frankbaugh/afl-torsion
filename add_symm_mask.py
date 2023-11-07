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
DIHEDRAL_PERIODIC_DIR = '/home/fmvb2/rds/rds-binding-a1NGKrlJtrw/dihedral_data/'

chi_pi_periodic = {
    'ALA': [0.0, 0.0, 0.0, 0.0],  # ALA
    'ARG': [0.0, 0.0, 0.0, 0.0],  # ARG
    'ASN': [0.0, 0.0, 0.0, 0.0],  # ASN
    'ASP': [0.0, 1.0, 0.0, 0.0],  # ASP
    'CYS': [0.0, 0.0, 0.0, 0.0],  # CYS
    'GLN': [0.0, 0.0, 0.0, 0.0],  # GLN
    'GLU': [0.0, 0.0, 1.0, 0.0],  # GLU
    'GLY': [0.0, 0.0, 0.0, 0.0],  # GLY
    'HIS': [0.0, 0.0, 0.0, 0.0],  # HIS
    'ILE': [0.0, 0.0, 0.0, 0.0],  # ILE
    'LEU': [0.0, 0.0, 0.0, 0.0],  # LEU
    'LYS': [0.0, 0.0, 0.0, 0.0],  # LYS
    'MET': [0.0, 0.0, 0.0, 0.0],  # MET
    'PHE': [0.0, 1.0, 0.0, 0.0],  # PHE
    'PRO': [0.0, 0.0, 0.0, 0.0],  # PRO
    'SER': [0.0, 0.0, 0.0, 0.0],  # SER
    'THR': [0.0, 0.0, 0.0, 0.0],  # THR
    'TRP': [0.0, 0.0, 0.0, 0.0],  # TRP
    'TYR': [0.0, 1.0, 0.0, 0.0],  # TYR
    'VAL': [0.0, 0.0, 0.0, 0.0],  # VAL
    'UNK': [0.0, 0.0, 0.0, 0.0],  # UNK
}

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


filelist = get_filelist()

for file in tqdm(filelist):
    protein_periodic_masks = []
    
    protein_id = file.replace('_protein.pdb', '')
    print(f'id: f{protein_id}')
    filepath = os.path.join(PDB_DIR, protein_id, file)
    parser=PDBParser(QUIET=True)
    structure=parser.get_structure(protein_id, filepath)
    
    residues = structure.get_residues()
    protein_res_torsions=[]
    protein_res_masks=[]
    
    for resi in residues:
        resi_name = resi.get_resname() 
        if resi_name not in aas: continue
        protein_periodic_masks.append(chi_pi_periodic[resi_name])
        
    with open(DIHEDRAL_DIR + protein_id + '.pickle', 'rb') as file:
        torsion_dict = pickle.load(file)
    
    torsion_dict['periodic_mask'] = protein_periodic_masks
    
    with open(DIHEDRAL_DIR + protein_id + '_periodic.pickle', 'wb') as wbf:
        pickle.dump(torsion_dict, file=wbf)