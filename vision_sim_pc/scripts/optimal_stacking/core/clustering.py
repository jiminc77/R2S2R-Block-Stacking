import numpy as np

class BlockClustering:
    """Cluster blocks based on Euclidean distance."""

    @staticmethod
    def cluster_blocks(block_poses, distance_threshold=0.10):
        """
        Cluster blocks into groups.
        
        Args:
            block_poses: List or np.ndarray of block positions [[x, y, z], ...]
            distance_threshold: Threshold in meters
            
        Returns:
            clusters: List of lists (indices of blocks in cluster)
        """
        positions = np.array(block_poses)
        n = len(positions)
        if n == 0:
            return []

        visited = np.zeros(n, dtype=bool)
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            current_cluster = [i]
            visited[i] = True
            queue = [i]

            while queue:
                curr_idx = queue.pop(0)
                curr_pos = positions[curr_idx]

                for j in range(n):
                    if not visited[j]:
                        dist = np.linalg.norm(positions[j] - curr_pos)
                        if dist <= distance_threshold:
                            visited[j] = True
                            current_cluster.append(j)
                            queue.append(j)
            
            clusters.append(current_cluster)

        return clusters

    @staticmethod
    def identify_tower_and_new_block(block_poses):
        """
        Identify the main tower (largest/closest) and new blocks.
        
        Returns:
            tower_indices: Indices of blocks in the tower
            other_indices: Indices of other blocks
        """
        clusters = BlockClustering.cluster_blocks(block_poses)
        
        if not clusters:
            return [], []

        cluster_metrics = []
        positions = np.array(block_poses)
        
        for cluster in clusters:
            size = len(cluster)
            cluster_positions = positions[cluster]
            centroid = np.mean(cluster_positions, axis=0)
            dist_origin = np.linalg.norm(centroid[:2])
            
            cluster_metrics.append({
                'indices': cluster,
                'size': size,
                'dist': dist_origin
            })
            
        # Sort by Size (Descending), then Distance (Ascending)
        cluster_metrics.sort(key=lambda x: (-x['size'], x['dist']))
        
        tower_indices = cluster_metrics[0]['indices']
        
        other_indices = []
        for c in cluster_metrics[1:]:
            other_indices.extend(c['indices'])
            
        return tower_indices, other_indices
