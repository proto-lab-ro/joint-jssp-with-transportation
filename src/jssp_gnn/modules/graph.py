import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from jssp_gnn.modules.GINEncoder import GINEncoder
from jssp_gnn.utils.tensor_ops import concat_node_graph_features


class SharedGraphFeatureExtractor(torch.nn.Module):
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

    def forward(self, observation):
        """
        Process graph observation from NewGraphEnv.

        Args:
            observation: TensorDict with observation["graph_data"] containing Data object(s)

        Returns:
            tuple: (node_embeddings, graph_embedding, batch_indices)
        """

        graph_data_list = []

        if isinstance(observation, Data):
            graph_data_list.append(observation)
        elif isinstance(observation, list):
            graph_data_list = observation
        else:
            raise ValueError(f"Unexpected observation type: {type(observation)}")

        batched_pyg_data = Batch.from_data_list(graph_data_list)

        node_embeddings = self.gnn(batched_pyg_data.x, batched_pyg_data.edge_index)

        graph_embedding = self.aggregate(node_embeddings, batched_pyg_data.batch)
        batch_size = batched_pyg_data.num_graphs
        num_nodes_per_graph = batched_pyg_data.x.size(0) // batch_size
        node_embeddings = node_embeddings.view(
            batch_size, num_nodes_per_graph, self.hidden_dim
        )

        if graph_embedding.shape[0] == 1:
            graph_embedding = graph_embedding.squeeze(0)
            node_embeddings = node_embeddings.squeeze(0)

        return node_embeddings, graph_embedding, batched_pyg_data.batch


class SB3LikeActor(torch.nn.Module):
    """
    Policy network that uses graph features to produce action logits.
    Uses node embeddings to compute logits for each job.
    """

    def __init__(self, shared_extractor, num_jobs, num_operations, forward_type="job"):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.num_jobs = num_jobs
        self.num_operations = num_operations
        self.forward_type = forward_type

        input_dim = shared_extractor.features_dim * 2

        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        for m in self.policy_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_to_operation_logits(self, node_embeddings, graph_embedding):
        num_operations = self.num_operations
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

    def _forward_to_job_logits(self, node_embeddings):
        is_batched = node_embeddings.dim() == 3
        if is_batched:
            batch_size, num_nodes, _ = node_embeddings.shape
            node_logits = self.policy_head(node_embeddings)

            num_ops_per_job = num_nodes // self.num_jobs

            node_logits = node_logits.view(
                batch_size, self.num_jobs, num_ops_per_job, 1
            )

            logits = node_logits.mean(dim=2)

            logits = logits.squeeze(-1)
        else:
            num_nodes = node_embeddings.shape[0]

            node_logits = self.policy_head(node_embeddings)

            num_ops_per_job = num_nodes // self.num_jobs

            node_logits = node_logits.view(self.num_jobs, num_ops_per_job, 1)
            logits = node_logits.mean(dim=1)

            logits = logits.squeeze(-1)

        return logits

    def forward(self, observation):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            logits: (batch_size, num_jobs) or (num_jobs,) action logits
        """
        node_embeddings, graph_embedding, _ = self.shared_extractor(observation)

        match self.forward_type:
            case "operation":
                return self._forward_to_operation_logits(
                    node_embeddings, graph_embedding
                )
            case "job":
                return self._forward_to_job_logits(node_embeddings)
            case _:
                raise ValueError(f"Unknown forward_type: {self.forward_type}")


class SB3LikeCritic(torch.nn.Module):
    """
    Value network that uses graph-level features to estimate state value.
    """

    def __init__(self, shared_extractor):
        super().__init__()
        self.shared_extractor = shared_extractor

        input_dim = shared_extractor.features_dim

        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        for m in self.value_head:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, observation):
        """
        Args:
            observation: Dict with 'graph_data' containing PyG Data object

        Returns:
            state_value: (batch_size, 1) estimated state value
        """
        _, graph_embedding, _ = self.shared_extractor(observation)

        state_value = self.value_head(graph_embedding)

        return state_value.unsqueeze(0) if state_value.dim() == 1 else state_value
