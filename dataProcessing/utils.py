import numpy as np

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3',
            'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O',
            'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}




aa_types= np.array((
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL'
))

def norm(u):
    """
    Computes the euclidean norm of a vector
    :param u: vector
    :return: euclidean norm of u
    """
    return np.sqrt(np.sum(u**2))

def gram_schmidt(u1, u2):
    """
    Orthonormalize a set of two vectors.
    :param u1: first non zero vector, unnormalized
    :param u2: second non zero vector, unormalized
    :return: orthonormal basis
    """
    e1 = u1/norm(u1)
    e2 = u2 - np.dot(u2, e1)*e1
    e2 /= norm(e2)
    return e1, e2

def get_orthonormal_basis(u1, u2):
    """
    Computes the local orthonormal frame basis based on the Nitrogen, C alpha and C beta atoms.
    :param u1: first non zero vector, unnormalized (here the bond vector between Nitrogen and C alpha carbon)
    :param u2: second non zero vector, unormalized  (here the bond vector between C beta and C alpha carbon)
    :return: A set of three orthonormal vectors
    """

    e1, e2 = gram_schmidt(u1, u2)
    e3 = np.cross(e1, e2)
    return np.array([e1, e2, e3]).T

def aa_one_hot(name):
    """
    Create the one-hot encoding vector for a specific amino-acid
    :param name: string, name of the amino acid in PDB format (e.g ALA for alanine)
    :return: np.array, type int, one hot encoding of the amino acid.
    """
    if name not in aa_types:
        raise ValueError("This amino-acid is unknown")

    return np.array(aa_types == name, dtype=int)


def compute_distance_matrix(locations):
    """
    Compute the distance matrix for all residue pairs.
    :param locations: numpy array of size (N_residues,3) of all C alpha positions
    :return: a symmetric numpy array of size (N_residues, N_residues) of pairwise distances
    """
    N_residues = len(locations)
    print(N_residues)
    distance_matrix = np.zeros((N_residues, N_residues))
    for i, pos in enumerate(locations):
        distance_matrix[i,i] = np.inf
        for j in range(i+1, N_residues):
            distance_matrix[i, j] = norm(pos - locations[j,:])

    distance_matrix += distance_matrix.T
    return distance_matrix


def find(ar, condition):
    """

    :param ar: array from whch we want the indexes
    :param condition: condition upon which we get the index
    :return: the set of indexes verifying the condition
    """
    return [i for i,val in enumerate(ar) if condition(val)]


def gaussian_rbf(x,y,sigma):
    return np.exp(-(1/2)*np.sum((x-y)**2)/sigma**2)