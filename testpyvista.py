import pyvista as pv
import numpy as np


class VTKWriter:
    def __init__(self, output_file):
        self.output_file = output_file

    def gmsh_to_vtk_cell_type(gmsh_type):
        gmsh_to_vtk = {
            1: 3,  # 2-node line -> VTK_LINE
            2: 5,  # 3-node triangle -> VTK_TRIANGLE
            3: 9,  # 4-node quadrangle -> VTK_QUAD
            4: 10,  # 4-node tetrahedron -> VTK_TETRA
            5: 12,  # 8-node hexahedron -> VTK_HEXAHEDRON
            6: 13,  # 6-node prism -> VTK_WEDGE
            7: 14,  # 5-node pyramid -> VTK_PYRAMID
            8: 21,  # 3-node second order line -> VTK_QUADRATIC_EDGE
            9: 22,  # 6-node second order triangle -> VTK_QUADRATIC_TRIANGLE
            10: 23,  # 9-node second order quadrangle -> VTK_QUADRATIC_QUAD
            11: 24,  # 10-node second order tetrahedron -> VTK_QUADRATIC_TETRA
            12: 25,  # 27-node second order hexahedron -> VTK_QUADRATIC_HEXAHEDRON
            13: 26,  # 18-node second order prism -> VTK_QUADRATIC_WEDGE
            14: 27,  # 14-node second order pyramid -> VTK_QUADRATIC_PYRAMID
        }
        return gmsh_to_vtk.get(gmsh_type, None)

    def write(self, node_coords, cells, cell_types, results):
        # vtk_cell_types = [VTKWriter.gmsh_to_vtk_cell_type(cell_type) for cell_type in cells]
        node_coords = node_coords.astype(np.float32)
        vtk_cell_types = np.array(
            [VTKWriter.gmsh_to_vtk_cell_type(cell_type) for cell_type in cell_types],
            dtype=np.uint8,
        )
        if vtk_cell_types is None:
            raise ValueError(f"Unsupported Gmsh cell type: {cell_types}")

        # PolyData用のセルタイプ
        polydata_cell_types = {
            3,
            5,
            9,
            7,
        }  # VTK_VERTEX, VTK_LINE, VTK_TRIANGLE, VTK_QUAD
        # UnstructuredGrid用のセルタイプ
        unstructured_grid_cell_types = {10, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27}

        # セルを PolyData と UnstructuredGrid 用に分ける
        polydata_cells = []
        polydata_types = []
        unstructured_grid_cells = []
        unstructured_grid_types = []

        for cell, vtk_type in zip(cells, vtk_cell_types):
            if vtk_type in polydata_cell_types:
                # polydata_cells.append(cell)
                polydata_cells.append([len(cell)] + list(cell))
                polydata_types.append(vtk_type)
            elif vtk_type in unstructured_grid_cell_types:
                unstructured_grid_cells.append(cell)
                unstructured_grid_types.append(vtk_type)
            else:
                raise ValueError(f"Unsupported VTK cell type: {vtk_type}")

        # PyVistaのUnstructuredGridを作成
        if polydata_cells:
            # polydata_cells = np.hstack([[len(cell)] + cell.tolist() for cell in polydata_cells])
            # grid = pv.PolyData(node_coords, polydata_cells)
            # grid = pv.PolyData(node_coords, cells)
            # polydata_cells = np.hstack(polydata_cells)
            # grid = pv.PolyData(node_coords, polydata_cells)
            # grid = pv.PolyData(node_coords, np.hstack([np.full((cells.shape[0], 1), 3), cells]).astype(np.int64))
            polydata_cells = np.hstack(polydata_cells)
            grid = pv.PolyData(node_coords, polydata_cells)
        elif unstructured_grid_cells:
            # unstructured_grid_cells = np.hstack([[len(cell)] + cell.tolist() for cell in unstructured_grid_cells])
            # grid = pv.UnstructuredGrid(unstructured_grid_cells, np.array(unstructured_grid_types), node_coords)
            # grid = pv.UnstructuredGrid(unstructured_grid_cells, vtk_cell_types, node_coords)
            grid = pv.UnstructuredGrid(cells, vtk_cell_types, node_coords)
        else:
            raise ValueError("No valid cells found for PolyData or UnstructuredGrid")
        # Add results as point data
        if results is not None:
            for key, value in results.items():
                grid.point_data[key] = value

        # Write to VTK file
        grid.save(self.output_file)


# Usage example
if __name__ == "__main__":
    # gmsh_file = "path/to/your/mesh.msh"
    vtk_file = "output.vtk"

    # mesh_reader = MeshReader(gmsh_file)
    # node_coords, triangles = mesh_reader.read_mesh()
    # points = np.array([
    #     [0, 0, 0],
    #     [1, 0, 0],
    #     [1, 1, 0],
    #     [0, 1, 0],
    #     [0.5, 0.5, 1]
    # ])

    # # セルの定義 (例: 四面体要素)
    # # セルの最初の数字は頂点の数を示す（4つの頂点の四面体）
    # cells = np.hstack([
    #     [4, 0, 1, 2, 4],
    #     [4, 0, 2, 3, 4]
    # ])

    # # セルのタイプ (例: gmsh type)
    # cell_types = np.array([4, 4])  # 2: 3-node triangle, 4: 4-node tetrahedron
    # scalars = np.array([1, 2, 3, 4, 5])
    # results = {
    #     "scalars": scalars
    # }
    # # results = {
    # #     "pressure": np.random.rand(node_coords.shape[0]),  # Example result data
    # #     "velocity": np.random.rand(node_coords.shape[0], 3)  # Example result data
    # # }

    # vtk_writer = VTKWriter(vtk_file)
    # # vtk_writer.write(node_coords, triangles, results)
    # vtk_writer.write(points,cells,cell_types, results)

    # ノード座標 (例)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    # セルの定義 (例: 三角形要素)
    cells = [[0, 1, 2], [0, 2, 3]]

    # セルのタイプ (例: gmsh type)
    cell_types = np.array([2, 2])  # 2: 3-node triangle

    # 結果の定義 (例: スカラー値)
    scalars = np.array([1, 2, 3, 4])
    results = {"scalars": scalars}

    vtk_writer = VTKWriter(vtk_file)
    vtk_writer.write(points, cells, cell_types, results)
