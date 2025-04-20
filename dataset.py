import os
import numpy as np
import torch

class SceneWiseDataLoader:
    def __init__(self, directory, device="cpu"):
        """
        Manual scene-wise data loader for full-scene ML training.

        Args:
            directory (str): Path to folder containing .npz scene files.
            device (str): Target device for tensors ("cpu" or "cuda").
        """
        self.file_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')])
        self.num_scenes = len(self.file_paths)
        self.device = device

        self.scenes = []
        self.labels = []
        self.num_points_per_scene = None

        self._load_scenes()

    def _load_scenes(self):
        for path in self.file_paths:
            data = np.load(path)
            points = data['points']  # (N_points, 3)
            labels = data['labels']  # (N_points,)

            # Consistency check: ensure all scenes have same point count
            if self.num_points_per_scene is None:
                self.num_points_per_scene = points.shape[0]
            else:
                assert points.shape[0] == self.num_points_per_scene, f"Scene {path} has inconsistent point count!"

            points = points - points.mean(axis=0, keepdims=True)
            points = points / (np.linalg.norm(points, axis=1).max() + 1e-6)

            self.scenes.append(points)
            self.labels.append(labels)

        print(f"✅ Loaded {self.num_scenes} scenes with {self.num_points_per_scene} points per scene.")

    def __len__(self):
        return self.num_scenes

    def get_batch(self, indices):
        """
        Returns a batch of scenes.

        Args:
            indices (list of int): Indices of scenes to load.

        Returns:
            points (Tensor): (B, N_points, 3)
            labels (Tensor): (B, N_points)
        """
        batch_points = [self.scenes[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]

        batch_points = torch.tensor(batch_points, dtype=torch.float32, device=self.device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)

        return batch_points, batch_labels

    def iterate_batches(self, batch_size, shuffle=True):
        """
        Generator to yield batches of scenes.

        Args:
            batch_size (int): Number of scenes per batch.
            shuffle (bool): Whether to shuffle scenes before batching.

        Yields:
            points (Tensor): (B, N_points, 3)
            labels (Tensor): (B, N_points)
        """
        indices = np.arange(self.num_scenes)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_scenes, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield self.get_batch(batch_indices)
