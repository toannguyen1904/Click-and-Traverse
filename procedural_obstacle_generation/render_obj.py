import sys
import trimesh

if __name__ == "__main__":
    mesh_file = "/home/tien/Code/Click-and-Traverse/data/assets/RandObs/D8G0L1O0S3/obs.obj"
    mesh = trimesh.load(mesh_file, force="mesh")
    print(f"Loaded: {mesh_file}")
    print(f"  Vertices : {len(mesh.vertices)}")
    print(f"  Faces    : {len(mesh.faces)}")
    print(f"  Bounds   : {mesh.bounds.tolist()}")

    mesh.show()
