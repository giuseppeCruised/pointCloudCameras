import torch
import open3d as o3d
import numpy as np
import argparse
import os
from main import add_noise_to_pointcloud

def visualize_segmented_scene(geometry, labels, frame, palette=None):
    """
    Visualizes a segmented point cloud or triangle mesh using Open3D.
    
    Parameters:
      geometry (o3d.geometry.PointCloud or o3d.geometry.TriangleMesh):
          The geometry you wish to display.
      labels (np.ndarray):
          A 1D numpy array of segmentation labels. For a point cloud, one label per point.
          For a mesh, one label per face.
      palette (dict, optional):
          A dictionary mapping each label to a color [R, G, B] (values in [0,1]). 
          If not provided, random colors will be generated.
    """
    # Create a palette if none is provided.
    if palette is None:
        unique_labels = np.unique(labels)
        palette = {label: np.random.rand(3).tolist() for label in unique_labels}
    
    # For a point cloud, assign colors to each point.
    if isinstance(geometry, o3d.geometry.PointCloud):
        print("was pointcloud")

        # Convert to tensor and add noise
        points = torch.tensor(np.asarray(geometry.points), dtype=torch.float32)
        points = add_noise_to_pointcloud(points)

        # Convert back to numpy for Open3D
        points = points.numpy()

        if len(points) != len(labels):
            raise ValueError("The number of points does not match the number of labels.")
        colors = np.array([palette[label] for label in labels])
        geometry.points = o3d.utility.Vector3dVector(points)
        geometry.colors = o3d.utility.Vector3dVector(colors)
    
    # For a mesh, assign colors per vertex based on the label of each face.
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        triangles = np.asarray(geometry.triangles)
        num_vertices = np.asarray(geometry.vertices).shape[0]
        vertex_colors = np.zeros((num_vertices, 3))
        counts = np.zeros(num_vertices)
        if len(triangles) != len(labels):
            raise ValueError("The number of triangles does not match the number of labels.")
        # For each triangle face, add its color to each vertex in the face.
        for face_idx, triangle in enumerate(triangles):
            color = np.array(palette[labels[face_idx]])
            for vertex_idx in triangle:
                vertex_colors[vertex_idx] += color
                counts[vertex_idx] += 1
        counts[counts == 0] = 1  # Avoid division by zero.
        vertex_colors = vertex_colors / counts[:, None]
        geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        raise TypeError("Geometry must be either a PointCloud or a TriangleMesh.")
    
    # Visualize the geometry.
    o3d.visualization.draw_geometries([geometry, frame])

def load_geometry(file_path, geom_type="auto"):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.npz':
        npzfile = np.load(file_path)
        if 'points' not in npzfile:
            raise ValueError("The .npz file does not contain a 'points' key.")
        points = npzfile['points']
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(points)

        if 'labels' in npzfile:
            labels = npzfile['labels']
            print(f"Loaded labels from {file_path}")
        else:
            num_points = points.shape[0]
            labels = np.zeros(num_points, dtype=int)
            print(f"No labels in {file_path}, using dummy labels.")

        return geometry, labels

    elif geom_type == "pcd" or (geom_type == "auto" and file_ext in [".pcd", ".ply", ".xyz"]):
        geometry = o3d.io.read_point_cloud(file_path)
        if not geometry.has_points():
            geometry = o3d.io.read_triangle_mesh(file_path)
    elif geom_type == "mesh" or (geom_type == "auto" and file_ext in [".obj", ".stl", ".off", ".ply"]):
        geometry = o3d.io.read_triangle_mesh(file_path)
    else:
        # Fallback: try point cloud first.
        geometry = o3d.io.read_point_cloud(file_path)
        if not geometry.has_points():
            geometry = o3d.io.read_triangle_mesh(file_path)
    return geometry

def main():
    parser = argparse.ArgumentParser(description="Segmented Scene Visualizer")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to the geometry file to visualize (point cloud or mesh).")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to a .npy file containing segmentation labels.")
    parser.add_argument("--type", type=str, default="auto", choices=["auto", "pcd", "mesh"],
                        help="Type of geometry to load. 'auto' infers from file extension.")

    parser.add_argument("--model", type=str, help="Path to trained model .pt file.")
    parser.add_argument("--reconstruct", action="store_true", help="Run model to denoise and visualize output.")
    args = parser.parse_args()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)

    if args.file:
        geometry, labels = load_geometry(args.file, args.type)
        
        if args.reconstruct and args.model:
            import torch
            from main import DenoisingModule  # <- replace with your model import

            points = np.asarray(geometry.points)
            points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
            points_tensor = add_noise_to_pointcloud(points_tensor)

            model = DenoisingModule(16)
            model.load_state_dict(torch.load(args.model, map_location="cpu"))
            model.eval()

            with torch.no_grad():
                reconstructed = model(points_tensor).squeeze(0).numpy()

            # visualize reconstructed point cloud
            recon_pcd = o3d.geometry.PointCloud()
            recon_pcd.points = o3d.utility.Vector3dVector(reconstructed)
            print("✅ Visualizing reconstructed point cloud")
            o3d.visualization.draw_geometries([recon_pcd, frame])

        else:
            visualize_segmented_scene(geometry, labels, frame)

if __name__ == "__main__":
    main()
