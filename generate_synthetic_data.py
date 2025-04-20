import os
import numpy as np
import open3d as o3d
import random

# --- Shape Creation Functions ---

def create_sphere_mesh(radius=1.0):
    """Creates an open3d sphere mesh."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    return mesh

def create_cube_mesh(width=2.0, height=2.0, depth=2.0):
    """Creates an open3d cube mesh centered at origin."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    # Center the mesh at the origin
    mesh.translate([-width / 2, -height / 2, -depth / 2])
    mesh.compute_vertex_normals()
    return mesh

def create_cylinder_mesh(radius=1.0, height=2.0):
    """Creates an open3d cylinder mesh."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    mesh.compute_vertex_normals()
    return mesh

def create_pyramid_mesh(base_size=2.0, height=1.5):
    """Creates an open3d pyramid mesh with a square base centered at origin."""
    half_base = base_size / 2.0
    vertices = [
        # Base vertices (z=0 plane)
        [-half_base, -half_base, 0], # 0
        [ half_base, -half_base, 0], # 1
        [ half_base,  half_base, 0], # 2
        [-half_base,  half_base, 0], # 3
        # Apex vertex
        [0, 0, height]              # 4
    ]
    triangles = [
        # Base (two triangles)
        [0, 1, 2],
        [0, 2, 3],
        # Sides
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    return mesh

# --- Scene Generation Logic ---

# Define available shapes, their creation functions, and labels
SHAPE_GENERATORS = {
    "sphere": (create_sphere_mesh, 0),
    "pyramid": (create_pyramid_mesh, 1),
    "cube": (create_cube_mesh, 2),
    "cylinder": (create_cylinder_mesh, 3),
}
SHAPE_NAMES = list(SHAPE_GENERATORS.keys())

def generate_scene(num_points_in_scene, random_seed=None, dense_sampling_factor=10):
    """
    Generates a synthetic 3D scene with four randomly chosen shapes.
    Points are sampled approximately uniformly from the combined surfaces.

    Parameters:
      num_points_in_scene (int): The target number of points for the final scene.
      random_seed (int, optional): Seed for numpy and random modules.
      dense_sampling_factor (int): Multiplier for initial dense sampling
                                   before final uniform selection.

    Returns:
      points (np.ndarray): Array of shape (N, 3) containing 3D coordinates.
                           N might be slightly less than num_points_in_scene
                           if shapes have zero area or sampling fails.
      labels (np.ndarray): 1D array of segmentation labels (0-3) for each point.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed) # Also seed the 'random' module for shape choices

    meshes_in_scene = []
    labels_in_scene = []
    surface_areas = []

    # 1. Randomly select and create 4 shapes
    for _ in range(4):
        shape_name = random.choice(SHAPE_NAMES)
        creator_func, label = SHAPE_GENERATORS[shape_name]

        # Create the mesh (using default parameters for simplicity here)
        # You could randomize parameters like radius, height etc. if needed
        mesh = creator_func()

        # Apply random translation
        translation = np.random.uniform(-4, 4, size=3) # Translate within a -4 to 4 box
        mesh.translate(translation)

        area = mesh.get_surface_area()
        if area > 1e-6: # Avoid issues with zero-area meshes
            meshes_in_scene.append(mesh)
            labels_in_scene.append(label)
            surface_areas.append(area)
        else:
            print(f"Warning: Generated mesh {shape_name} has near-zero area. Skipping.")


    if not meshes_in_scene:
        print("Warning: No valid meshes generated for this scene. Returning empty arrays.")
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    # 2. Calculate dense sampling points based on surface area
    total_surface_area = sum(surface_areas)
    total_dense_points_needed = num_points_in_scene * dense_sampling_factor

    all_dense_points = []
    all_dense_labels = []

    if total_surface_area < 1e-6:
         print("Warning: Total surface area is near-zero. Cannot sample points.")
         return np.empty((0, 3)), np.empty((0,), dtype=int)

    # 3. Perform dense sampling on each mesh proportionally
    for mesh, label, area in zip(meshes_in_scene, labels_in_scene, surface_areas):
        num_dense_samples = int(np.ceil((area / total_surface_area) * total_dense_points_needed))
        if num_dense_samples <= 0:
            continue

        try:
            # Use sample_points_uniformly for surface sampling
            pcd = mesh.sample_points_uniformly(number_of_points=num_dense_samples)
            points = np.asarray(pcd.points)
            if points.shape[0] > 0:
                all_dense_points.append(points)
                all_dense_labels.append(np.full(points.shape[0], label, dtype=int))
        except Exception as e:
            print(f"Warning: Could not sample points from a mesh: {e}")


    if not all_dense_points:
        print("Warning: Dense sampling resulted in no points. Returning empty arrays.")
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    # 4. Combine all densely sampled points
    combined_dense_points = np.vstack(all_dense_points)
    combined_dense_labels = np.concatenate(all_dense_labels)

    # 5. Uniformly sample the final number of points from the dense pool
    num_available_dense_points = combined_dense_points.shape[0]
    num_final_points = min(num_points_in_scene, num_available_dense_points)

    if num_final_points <= 0:
        print("Warning: No points available for final sampling. Returning empty arrays.")
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    # Use np.random.choice for efficient uniform sampling without replacement
    final_indices = np.random.choice(num_available_dense_points, size=num_final_points, replace=False)

    final_points = combined_dense_points[final_indices]
    final_labels = combined_dense_labels[final_indices]

    # Optional: Shuffle the final points (although selection was random)
    shuffle_idx = np.arange(final_points.shape[0])
    np.random.shuffle(shuffle_idx)
    final_points = final_points[shuffle_idx]
    final_labels = final_labels[shuffle_idx]

    return final_points, final_labels

# --- Dataset Generation Function ---

def generate_dataset(num_scenes, output_dir, num_points_per_scene, seed_offset=0):
    """
    Generates multiple scenes and saves each one as an .npz file in output_dir.

    Parameters:
      num_scenes (int): Number of scenes to generate.
      output_dir (str): Directory where scenes will be saved.
      num_points_per_scene (int): Target number of points in each generated scene.
      seed_offset (int): Offset for the random seed for reproducibility/splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_scenes):
        scene_seed = i + seed_offset
        points, labels = generate_scene(num_points_in_scene=num_points_per_scene,
                                        random_seed=scene_seed)

        if points.shape[0] == 0:
             print(f"Skipping scene {i} due to generation errors.")
             continue

        output_file = os.path.join(output_dir, f"scene_{i}.npz")
        np.savez(output_file, points=points, labels=labels)
        print(f"Saved scene {i} ({points.shape[0]} points) to {output_file}")

# --- Main Execution ---

def main():
    # Hardcoded parameters
    NUM_TRAIN_SCENES = 256  # Number of training scenes
    NUM_TEST_SCENES = 20   # Number of testing scenes
    NUM_POINTS_PER_SCENE = 2048 # Target number of points per scene
    TRAIN_DIR = "train_random_shapes"
    TEST_DIR = "test_random_shapes"

    print("Generating training dataset...")
    # Use seed_offset=0 for training set
    generate_dataset(NUM_TRAIN_SCENES, TRAIN_DIR,
                     num_points_per_scene=NUM_POINTS_PER_SCENE, seed_offset=0)

    print("\nGenerating testing dataset...")
    # Use a different seed_offset for the test set to ensure different scenes
    generate_dataset(NUM_TEST_SCENES, TEST_DIR,
                     num_points_per_scene=NUM_POINTS_PER_SCENE, seed_offset=NUM_TRAIN_SCENES)

    print("\nDataset generation complete.")

if __name__ == "__main__":
    # Ensure you have open3d and numpy installed:
    # pip install open3d numpy
    main()
