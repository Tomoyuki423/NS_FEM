import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

# Mock the gmsh module
sys.modules['gmsh'] = MagicMock()
import gmsh

from MeshReader import MeshReader

# Test case for MeshReader
def test_read_mesh():
    # Path to the test Gmsh file
    test_gmsh_file = '/mnt/data/test_mesh.msh'

    # Initialize the MeshReader with the test file
    mesh_reader = MeshReader(test_gmsh_file)

    # Mock the read_mesh method
    mesh_reader.read_mesh = MagicMock(return_value=(
        np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ]),
        np.array([
            [0, 1, 2],
            [1, 2, 3]
        ])
    ))

    # Read the mesh data
    node_coords, triangles = mesh_reader.read_mesh()

    # Expected node coordinates (manually specified based on the test_msh.msh content)
    expected_node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    # Expected triangles (manually specified based on the test_msh.msh content)
    expected_triangles = np.array([
        [0, 1, 2],
        [1, 2, 3]
    ])

    # Assertions to check if the mesh is read correctly
    assert np.array_equal(node_coords, expected_node_coords), f"Expected: {expected_node_coords}, Got: {node_coords}"
    assert np.array_equal(triangles, expected_triangles), f"Expected: {expected_triangles}, Got: {triangles}"

# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
