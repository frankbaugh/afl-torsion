import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import calc_dihedral
import numpy as np
import pickle
import os, traceback
from tqdm import tqdm

"""
DOES three things to torsion data
1) Convert pdb ID to Shengyu ID
2) Convert each AA name into a single value [0->20] (this may not be necessary)
"""


PDB_DIR='/rds/project/rds-a1NGKrlJtrw/dyna_mol/v2020-other-PL/'
DIHEDRAL_DIR= '/home/fmvb2/rds/rds-binding-a1NGKrlJtrw/dihedral_data/'
CSV_PATH = '/rds/project/rds-a1NGKrlJtrw/dyna_mol/pdb_1024_processed_trunc.csv'

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
    for subdir, dirs, files in os.walk(DIHEDRAL_DIR):
        for file in files:
            if '.pickle' in file:
                filelist.append(file)
#print(filelist)
    return filelist


filelist = get_filelist()


for file in tqdm(filelist):

    protein_id = file.replace('.pickle', '')
    
    with open(DIHEDRAL_DIR + protein_id + '.pickle', 'rb') as file:
        torsion_dict = pickle.load(file)
    
    # torsion_dict['aa_index'] = [aas.index(aa) for aa in torsion_dict['aa_list']]
    torsion_array = np.array(torsion_dict['dihedrals'])

    pdb = pd.read_csv(CSV_PATH, usecols=['pdb', 'id'])
    
    shengyu_id = pdb.loc[pdb['pdb'] == protein_id, 'id'].values[0]
    # print(f"pdb: {protein_id}   shengyu: {shengyu_id}")
    
    np.save(DIHEDRAL_DIR + shengyu_id + '_dihedrals.npy', torsion_array)