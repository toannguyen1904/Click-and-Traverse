import argparse
import trimesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", required=True)
    args = parser.parse_args()
    mesh_file = args.mesh_file
    mesh = trimesh.load(mesh_file, force="mesh")
    print(f"Loaded: {mesh_file}")
    print(f"  Vertices : {len(mesh.vertices)}")
    print(f"  Faces    : {len(mesh.faces)}")
    print(f"  Bounds   : {mesh.bounds.tolist()}")

    mesh.show()
