import sys
sys.path.append('/home/fmvb2/')
sys.path.append('/home/fmvb2/openfold/')
import openfold
from torch_af_ligand.afl.data import features

PDB_DIR='/rds/project/rds-a1NGKrlJtrw/dyna_mol/v2020-other-PL/3sm2/'
file = '3sm2_protein.pdb'

pdb_feats = features.pdb_feat_loader(PDB_DIR + file, chain_id=None, is_distillation=False)
print(pdb_feats)