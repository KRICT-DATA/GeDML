import numpy
import pandas
import torch
from rdkit import Chem
from mendeleev import get_table
from sklearn.preprocessing import scale
from torch_geometric.data import Data


atm_prop_names = ['atomic_number', 'atomic_radius', 'atomic_volume', 'boiling_point', 'density',
                  'dipole_polarizability', 'electron_affinity', 'evaporation_heat', 'fusion_heat',
                  'lattice_constant', 'melting_point', 'period', 'specific_heat', 'thermal_conductivity',
                  'vdw_radius', 'covalent_radius_cordero', 'covalent_radius_pyykko', 'en_pauling',
                  'en_allen', 'heat_of_formation', 'vdw_radius_uff', 'vdw_radius_mm3', 'abundance_crust',
                  'abundance_sea', 'en_ghosh', 'vdw_radius_alvarez',
                  'c6_gb', 'atomic_weight', 'atomic_weight_uncertainty', 'atomic_radius_rahm']

num_atm_feats = len(atm_prop_names)


def read_atom_prop():
    tb_atm_props = get_table('elements')
    arr_atm_nums = numpy.array(tb_atm_props['atomic_number'], dtype=numpy.int)
    arr_atm_props = numpy.nan_to_num(numpy.array(tb_atm_props[atm_prop_names], dtype=numpy.float))
    arr_atm_props = scale(arr_atm_props)
    atm_props_mat = {arr_atm_nums[i]: arr_atm_props[i, :] for i in range(0, arr_atm_nums.shape[0])}

    return atm_props_mat


def read_dataset(file_name, target_idx=1):
    samples = []
    data_mat = numpy.array(pandas.read_csv(file_name))
    smiles = data_mat[:, 0]
    target = data_mat[:, target_idx]

    for i in range(0, data_mat.shape[0]):
        mol_graph = smiles_to_mol_graph(smiles[i], target[i], idx=i)

        if mol_graph is not None:
            samples.append(mol_graph)

        if (i + 1) % 50 == 0:
            print('Loading ' + str(i + 1) + 'th molecule was completed.')

    return samples, list(smiles)


def smiles_to_mol_graph(smiles, target, idx=-1):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        atom_feat_mat = numpy.empty([mol.GetNumAtoms(), atomic_props.get(1).shape[0]])

        ind = 0
        for atom in mol.GetAtoms():
            atom_feat_mat[ind, :] = atomic_props.get(atom.GetAtomicNum())
            ind = ind + 1

        bonds = list()
        for i in range(0, mol.GetNumAtoms()):
            for j in range(0, mol.GetNumAtoms()):
                if adj_mat[i, j] == 1:
                    bonds.append([i, j])
        bonds = torch.tensor(bonds, dtype=torch.long).cuda()

        if bonds.shape[0] == 0:
            return None

        mol_graph = Data(x=torch.tensor(atom_feat_mat, dtype=torch.float).cuda(),
                         edge_index=bonds.t().contiguous(),
                         y=torch.tensor(target, dtype=torch.float).view(-1, 1).cuda(),
                         idx=idx)

        return mol_graph
    except:
        print(smiles + ' could not be converted to molecular graph due to the internal errors of RDKit')
        return None


atomic_props = read_atom_prop()
