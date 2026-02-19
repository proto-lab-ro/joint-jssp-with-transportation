import torch
from torch_geometric.nn import global_mean_pool

from jssp_core.modules.nets import ActorPolicyHead, CriticValueHead
from jssp_gnn.modules.GINEncoder import GINEncoder
from jssp_gnn.utils.tensor_ops import concat_node_graph_features


class SharedGraphFeatureExtractorMatrix(torch.nn.Module):
    """
    Extracts graph features using a GINEncoder.
    Produces:
    - node_embeddings: embeddings for all nodes (for policy logits)
    - graph_embedding: pooled graph representation (for value function)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        k_layers=3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.features_dim = hidden_dim
        self.gnn = GINEncoder(input_dim, hidden_dim, k_layers)
        self.aggregate = global_mean_pool

    def _batch_graphs_from_tensor(self, node_feats, edge_index):
        """
        Manually batch multiple graphs from tensor format by creating a disjoint union.

        Args:
            node_feats: Batched node features tensor [batch_size, num_nodes, feature_dim]
            edge_index: Edge indices tensor [batch_size, 2, num_edges]
                       or [2, num_edges] if same structure for all graphs

        Returns:
            tuple: (batched_node_feats, batched_edge_index, batch_indices)
                - batched_node_feats: [total_nodes, feature_dim]
                - batched_edge_index: [2, total_edges]
                - batch_indices: [total_nodes] indicating which graph each node belongs to
        """
        batch_size = node_feats.size(0)
        num_nodes_per_graph = node_feats.size(1)
        feature_dim = node_feats.size(2)

        batched_node_feats = node_feats.view(-1, feature_dim)

        if edge_index.dim() == 3:
            num_edges = edge_index.size(2)
            batched_edge_indices = []

            for graph_idx in range(batch_size):
                offset = graph_idx * num_nodes_per_graph
                shifted_edge_index = edge_index[graph_idx] + offset
                batched_edge_indices.append(shifted_edge_index)

            batched_edge_index = torch.cat(batched_edge_indices, dim=1)
        else:
            num_edges = edge_index.size(1)
            batched_edge_indices = []

            for graph_idx in range(batch_size):
                offset = graph_idx * num_nodes_per_graph
                shifted_edge_index = edge_index + offset
                batched_edge_indices.append(shifted_edge_index)

            batched_edge_index = torch.cat(batched_edge_indices, dim=1)

        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=node_feats.device),
            num_nodes_per_graph,
        )

        return batched_node_feats, batched_edge_index, batch_indices

    def forward(self, node_feats, edge_index):
        """
        Process graph observation from NewGraphEnv.

        Args:
            node_feats: Node features tensor
                - Batched: [..., num_nodes, feature_dim]
            edge_index: Edge indices tensor
                - Batched: [..., 2, num_edges]

        Returns:
            tuple: (node_embeddings, graph_embedding, batch_indices)
        """

        original_batch_shape = node_feats.shape[:-2]
        is_batched = len(original_batch_shape) > 0

        if is_batched:
            num_nodes = node_feats.shape[-2]
            feature_dim = node_feats.shape[-1]

            node_feats_flat = node_feats.flatten(0, -3)

            if edge_index.dim() > 2:
                edge_index_flat = edge_index.flatten(0, -3)
            else:
                edge_index_flat = edge_index

            (
                node_feats_batched,
                edge_index_batched,
                batch_indices,
            ) = self._batch_graphs_from_tensor(node_feats_flat, edge_index_flat)

            node_embeddings_flat = self.gnn(node_feats_batched, edge_index_batched)

            graph_embedding_flat = self.aggregate(
                node_embeddings_flat, batch=batch_indices
            )

            node_embeddings = node_embeddings_flat.view(
                *original_batch_shape, num_nodes, self.hidden_dim
            )
            graph_embedding = graph_embedding_flat.view(
                *original_batch_shape, self.hidden_dim
            )

        else:
            batch_indices = torch.zeros(
                node_feats.size(0), dtype=torch.long, device=node_feats.device
            )
            node_embeddings = self.gnn(node_feats, edge_index)
            graph_embedding = self.aggregate(node_embeddings, batch=batch_indices)

            if graph_embedding.shape[0] == 1:
                graph_embedding = graph_embedding.squeeze(0)
                node_embeddings = node_embeddings.squeeze(0)

        return node_embeddings, graph_embedding, batch_indices


class SB3LikeActorMatrix(torch.nn.Module):
    """
    Policy network that uses graph features to produce action logits.
    Uses node embeddings to compute logits for each job.
    """

    def __init__(self, shared_extractor, forward_type="job"):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.forward_type = forward_type

        input_dim = shared_extractor.features_dim * 2
        self.policy_head = ActorPolicyHead(input_dim)

    def _forward_to_operation_logits(self, node_embeddings, graph_embedding, mask):
        num_operations = mask.shape[-1]
        if node_embeddings.dim() == 3:
            operation_embeddings = node_embeddings[:, :num_operations, :]
        elif node_embeddings.dim() == 2:
            operation_embeddings = node_embeddings[:num_operations, :]
        else:
            raise ValueError(
                f"Unexpected node_embeddings dimension: {node_embeddings.dim()}"
            )
        x = concat_node_graph_features(operation_embeddings, graph_embedding)

        logits = self.policy_head(x)
        logits = logits.squeeze(-1)

        return logits

    def _forward_to_job_logits(self, node_embeddings, mask):
        num_operations = mask.shape[-1]

        if node_embeddings.dim() == 3:
            operation_embeddings = node_embeddings[:, :num_operations, :]
        elif node_embeddings.dim() == 2:
            operation_embeddings = node_embeddings[:num_operations, :]
        else:
            raise ValueError(
                f"Unexpected node_embeddings dimension: {node_embeddings.dim()}"
            )

        is_batched = operation_embeddings.dim() == 3
        if is_batched:
            batch_size, num_nodes, _ = operation_embeddings.shape
            node_logits = self.policy_head(operation_embeddings)

            num_ops_per_job = num_nodes // self.num_jobs

            node_logits = node_logits.view(
                batch_size, self.num_jobs, num_ops_per_job, 1
            )

            logits = node_logits.mean(dim=2)

            logits = logits.squeeze(-1)
        else:
            num_nodes = operation_embeddings.shape[0]

            node_logits = self.policy_head(operation_embeddings)

            num_ops_per_job = num_nodes // self.num_jobs

            node_logits = node_logits.view(self.num_jobs, num_ops_per_job, 1)
            logits = node_logits.mean(dim=1)

            logits = logits.squeeze(-1)

        return logits

    def forward(self, node_feats, edge_index, mask=None):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            logits: (batch_size, num_jobs) or (num_jobs,) action logits
        """
        node_embeddings, graph_embedding, _ = self.shared_extractor(
            node_feats, edge_index
        )

        match self.forward_type:
            case "operation":
                return self._forward_to_operation_logits(
                    node_embeddings, graph_embedding, mask
                )
            case "job":
                return self._forward_to_job_logits(node_embeddings, mask)
            case _:
                raise ValueError(f"Unknown forward_type: {self.forward_type}")


class SB3LikeCriticMatrix(torch.nn.Module):
    """
    Value network that uses graph-level features to estimate state value.
    """

    def __init__(self, shared_extractor):
        super().__init__()
        self.shared_extractor = shared_extractor

        input_dim = shared_extractor.features_dim

        self.value_head = CriticValueHead(input_dim)

    def forward(self, node_feats, edge_index):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            state_value: (batch_size, 1) estimated state value
        """
        _, graph_embedding, _ = self.shared_extractor(node_feats, edge_index)

        state_value = self.value_head(graph_embedding)

        return state_value.unsqueeze(0) if state_value.dim() == 1 else state_value
