import json
import utils
import numpy as np
from collections import OrderedDict
from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.transform import Rotation

K_neighbours = 1550
rbf_sigma = np.linspace(0, 20, 17)[1:]


file = "../data/ranked_0_round1.pdb"
out_file = "../data/features.json"
parser = PDBParser(PERMISSIVE=0)

structure = parser.get_structure("A", file)

def get_positions(residue, name):
    x = residue["CA"].get_coord()
    y = residue["N"].get_coord()
    if name == "GLY":
        z = residue["C"].get_coord()
        return x,y,z

    z = residue["C"].get_coord()
    return x,y,z


def get_node_features(structure):
    residues_dict = OrderedDict()
    nodes_features = OrderedDict()

    N_residue = 0
    residues_indexes = []
    c_alpha_positions = []

    for model in structure:
        for chain in model:
            for residue in chain:
                print(N_residue)
                residues_indexes.append(N_residue)
                name = residue.get_resname()
                x,y,z = get_positions(residue, name)
                c_alpha_positions.append(x)
                local_frame = utils.get_orthonormal_basis(y-x, x - z)

                residues_dict[N_residue] = {"x":x, "y":y, "z":z, "name":name}
                nodes_features[N_residue] = {"x":x, "frame":local_frame, "aa_type": utils.aa_one_hot(name)}
                N_residue += 1

    return residues_dict,nodes_features, np.array(residues_indexes), np.array(c_alpha_positions)


def get_edge_features(residues_indexes ,pairwise_distances, nodes_features):
    edges_features = {}
    for n_residue in residues_indexes:
        print(n_residue/len(residues_indexes))
        sorted_residues = sorted(zip(pairwise_distances[n_residue, :], residues_indexes))
        neighbours_distances, neighbours_indexes = zip(*sorted_residues)

        edges_features[n_residue] = {}
        for neighb, distance in zip(neighbours_indexes[:K_neighbours], neighbours_distances[:K_neighbours]):
            distance_feature = np.array([utils.gaussian_rbf(nodes_features[n_residue]["x"], nodes_features[neighb]["x"],
                                                            sig) for sig in rbf_sigma])
            coordinates_local_frame = np.matmul(nodes_features[n_residue]["frame"].T,
                                                (nodes_features[neighb]["x"] - nodes_features[n_residue][
                                                    "x"]) / distance)

            relative_orientation_matrix = np.matmul(nodes_features[n_residue]["frame"].T,
                                                    nodes_features[neighb]["frame"])
            rot = Rotation.from_matrix(relative_orientation_matrix)
            relative_orientation_quaternion = rot.as_quat()

            edges_features[n_residue][neighb] = {"distances": distance_feature,
                                                 "coordinates_local_frame": coordinates_local_frame,
                                                 "quaternions": relative_orientation_quaternion}

    return edges_features


def get_edge_index(residues_indexes, edges_features):
    edge_index = []
    for i in residues_indexes:
        source_nodes = np.array([k for k in edges_features[i].keys()], dtype=int)
        target_node = np.ones(len(source_nodes), dtype=int) * i
        if len(edge_index) == 0:
            edge_index = np.vstack((source_nodes, target_node))
            continue

        edge_index_add = np.vstack((source_nodes, target_node))
        edge_index = np.hstack((edge_index, edge_index_add))

    return edge_index

#def get_data_x(residues_indexes, nodes_features):



residues_dict,nodes_features,residues_indexes, c_alpha_positions = get_node_features(structure)
pairwise_distances = utils.compute_distance_matrix(c_alpha_positions)
edges_features = get_edge_features(residues_indexes, pairwise_distances, nodes_features)
all_features = {"nodes_features":nodes_features, "edges_features":edges_features, "N_residues":len(residues_indexes)}


np.save(out_file, all_features)









