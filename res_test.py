from calculate_dihedrals import 

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

filelist = get_filelist()[0:1]
for file in tqdm(filelist):
    protein_id = file.replace('_protein.pdb', '')
    print(f'id: f{protein_id}')
    filepath = os.path.join(PDB_DIR, protein_id, file)
    print(filepath)
    parser=PDBParser()
    structure=parser.get_structure(protein_id, filepath)
    residues_shengyu = []
    residues_mine = structure.get_residues() # Is this the same as below?
    for chain in list(structure[0].get_chains()):
        residues_shengyu += list(chain.get_residues())

print("My residues:")
print(residues_mine[0:20])

print()
print("shengyu residues")
print(residues_shengyu[0:20])