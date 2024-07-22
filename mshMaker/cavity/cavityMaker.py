import gmsh
import numpy as np

def generate_cavity_mesh():
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("cavity")

    # Define the cavity geometry
    lc = 1e-1  # Characteristic length
    corner_points = [
        gmsh.model.geo.addPoint(0, 0, 0, lc),
        gmsh.model.geo.addPoint(1, 0, 0, lc),
        gmsh.model.geo.addPoint(1, 1, 0, lc),
        gmsh.model.geo.addPoint(0, 1, 0, lc)
    ]

    # Define lines between corner points to form the cavity
    lines = [
        gmsh.model.geo.addLine(corner_points[0], corner_points[1]),
        gmsh.model.geo.addLine(corner_points[1], corner_points[2]),
        gmsh.model.geo.addLine(corner_points[2], corner_points[3]),
        gmsh.model.geo.addLine(corner_points[3], corner_points[0])
    ]

    # Create a loop and a plane surface from the lines
    loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Synchronize to finalize the geometry definition
    gmsh.model.geo.synchronize()

    # Define physical groups (optional)
    gmsh.model.addPhysicalGroup(2, [surface], name="Cavity")
    gmsh.model.setPhysicalName(2, 1, "Cavity Surface")
    gmsh.model.addPhysicalGroup(1, [lines[0]], name="Inlet")
    gmsh.model.addPhysicalGroup(1, [lines[1]], name="Top Wall")
    gmsh.model.addPhysicalGroup(1, [lines[2]], name="Outlet")
    gmsh.model.addPhysicalGroup(1, [lines[3]], name="Bottom Wall")

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh to a file
    gmsh.write("cavity.msh")

    # Finalize Gmsh
    gmsh.finalize()

if __name__ == "__main__":
    generate_cavity_mesh()
