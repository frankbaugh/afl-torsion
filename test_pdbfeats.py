import sys
sys.path.append('/home/fmvb2/torch_af_ligand/')
from data.features import pdb_feat_loader

PDB_DIR='/rds/project/rds-a1NGKrlJtrw/dyna_mol/v2020-other-PL/3sm2/'
file = '3sm2_protein.pdb'

pdb_feats = pdb_feat_loader(PDB_DIR + file, chain_id=None, is_distillation=False)
print(pdb_feats)