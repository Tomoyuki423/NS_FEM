import gmsh
import numpy as np


class MeshReader:
    def __init__(self, gmsh_file):
        self.gmsh_file = gmsh_file
        gmsh.initialize()
        gmsh.open(self.gmsh_file)

    def read_mesh(self):
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()
        print("Node tags: ", node_tags)
        print("Node coords: ", node_coords)
        print("Element types: ", element_types)
        print("Element tags: ", element_tags)

        # Reshape node_coords into (node_count, 3)
        node_coords = np.array(node_coords).reshape(-1, 3)

        # Initialize an empty list for triangles
        triangles = []

        # Loop through element types to find 2D triangles
        for elem_type, elem_nodes in zip(element_types, element_nodes):
            if elem_type == 2:  # 2D triangle
                triangles.extend(elem_nodes)

        # Reshape triangles into (-1, 3)
        triangles = np.array(triangles).reshape(-1, 3)

        gmsh.finalize()

        return node_coords, triangles


def main():
    # Initialize input data
    meshreader = MeshReader("t1.msh")
    node_coords, triangles = meshreader.read_mesh()
    print(node_coords)
    print(node_coords.shape)
    print(triangles)
    print(triangles.shape)


if __name__ == "__main__":
    main()
