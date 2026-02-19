import torch

from jssp_core.domain import EnvironmentType
from jssp_core.modules.nets import SmallActorPolicyHead


class MlpAgvActor(torch.nn.Module):
    """
    Policy network that uses graph features to produce action logits.
    Uses node embeddings to compute logits for each job.
    """

    def __init__(self, input_dim):
        super().__init__()

        self.agent_type = EnvironmentType.MULTI_AGENT
        self.policy_head = SmallActorPolicyHead(input_dim)

    def forward(self, agv_embeddings):
        logits = self.policy_head(agv_embeddings)
        batch_size = agv_embeddings.size(0)
        num_agvs = agv_embeddings.shape[-2]
        if agv_embeddings.dim() == 3:
            batch_size = agv_embeddings.shape[0]
            logits = logits.reshape(batch_size, num_agvs)

        if agv_embeddings.dim() == 4:
            num_batches = agv_embeddings.shape[0]
            batch_size = agv_embeddings.shape[1]

            logits = logits.reshape(num_batches, batch_size, num_agvs)
        return logits
