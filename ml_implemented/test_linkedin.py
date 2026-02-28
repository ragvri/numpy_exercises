"""
The Problem: Two-Tower Retrieval for Job RecommendationsAt LinkedIn,
we need to retrieve the top $k$ relevant jobs for a user from a pool of millions of job postings.
A standard approach is a Two-Tower (Dual Encoder) architecture, where we independently encode
User features and Job features into a shared embedding space.

Your Task (Time: 20-30 mins):Using PyTorch, implement a JobRecommender class
with the following requirements

:Architecture:User Tower: Takes in a dictionary of features
(e.g., member_id embedding, dense skill_vector).

Job Tower: Takes in a dictionary of features (e.g., job_id embedding, dense title_vector).

Output: The dot product (similarity) between the normalized User and Job embeddings.

Custom Loss Function (Crucial):Standard Binary Cross Entropy is often too slow for
training retrieval models because it requires explicit negative sampling pipelines.
Implement In-Batch Negative Sampling (also known as In-Batch Softmax).

For a batch of $N$ (User, Positive Job) pairs,
treat the other $N-1$ jobs in the batch as negatives for each user.

Please write the PyTorch code for the Model and the Custom Loss function.
"""

import torch
from torch import Tensor
from torch import nn
from jaxtyping import Float
import torch.distributed as dist


class UserTower(nn.Module):
    def __init__(self, user_feats: dict, output_d: int):
        super().__init__()
        self.feature_processor = nn.ModuleDict()
        total_dims = 0
        for feature_name in user_feats:
            config = user_feats[feature_name]
            type = config["type"]
            if type == "id":
                layer = nn.Embedding(
                    num_embeddings=config["vocab_size"],
                    embedding_dim=config["dimension"],
                )
                total_dims += config["dimension"]
            elif type == "dense":
                layer = nn.Identity()
                total_dims += config["dimension"]
            self.feature_processor[feature_name] = layer

        self.rms = nn.RMSNorm(
            normalized_shape=[
                total_dims,
            ]
        )
        self.linear = nn.Linear(in_features=total_dims, out_features=output_d)

    def forward(self, feats: dict[str, Float[Tensor, "b dims"]]):
        embeds = []
        for feature in feats:
            embedding = self.feature_processor[feature]
            embeds.append(embedding)
        concat = torch.concatenate(embeds, dim=1)  # n x dk
        rms = self.rms(concat)
        user_embedding = self.linear(rms)
        return user_embedding


class JobTower(nn.Module):
    def __init__(self, output_d: int, job_feats_config: dict):
        super().__init__()
        self.output_d = output_d
        self.feature_processor = nn.ModuleDict()
        total_dims = 0
        for feautre in job_feats_config:
            config = job_feats_config[feautre]
            if config["type"] == "id":
                layer = nn.Embedding(
                    num_embeddings=config["vocab_size"], embedding_dim=config["dim"]
                )
                total_dims += config["dim"]
            else:
                layer = nn.Identity()
                total_dims += config["dim"]
            self.feature_processor[feautre] = layer
        self.rms = nn.RMSNorm(
            normalized_shape=[
                total_dims,
            ]
        )
        self.linear = nn.Linear(in_features=total_dims, out_features=output_d)

    def forward(self, feats: dict[str, Float[Tensor, "b d"]]):
        embeddings = []
        for feature in feats:
            embedding = self.feature_processor["feats"]
            embeddings.append(embedding)
        concat = torch.concatenate(embeddings, dim=1)  # n x dk
        rms = self.rms(concat)
        job_embedding = self.linear(rms)
        return job_embedding


class JobRecommender(nn.Module):
    def __init__(
        self,
        output_d: int,
        user_feats_config: dict,
        job_feats_config: dict,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature_log = nn.Parameter(
            torch.log(1 / torch.tensor(temperature))
        )  # needs to be a positive. So we do log and then take exp later to make sure temp remains positive
        self.output_d = output_d
        self.job_tower = JobTower(output_d, job_feats_config=job_feats_config)
        self.user_tower = UserTower(user_feats=user_feats_config, output_d=output_d)
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        user_feats: dict[str, Float[Tensor, "b d"]],
        job_feats: dict[str, Float[Tensor, "b d"]],
    ):
        user_embedding = self.user_tower(user_feats)  # b ,d
        # normalize
        user_embedding = user_embedding / torch.sqrt(
            torch.sum(user_embedding**2, dim=1, keepdim=True)
        )
        job_embedding = self.job_tower(job_feats)  # b, d
        job_embedding = job_embedding / torch.sqrt(
            torch.sum(job_embedding**2, dim=1, keepdim=True)
        )
        if dist.is_initialized():
            # need to gather job embeds from other gpus
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            all_job_embeds = [
                torch.zeros_like(job_embedding) for _ in range(world_size)
            ]

            dist.all_gather(
                all_job_embeds, job_embedding
            )  # order is deterministic by rank

            all_job_embeds[rank] = (
                job_embedding  # all gather creates copies so there is no grad attached. we attach local gradients back for backprop
            )
            global_job_embeds = torch.concat(all_job_embeds, dim=0)  # B x num gpus, d

            logits = user_embedding @ global_job_embeds.T
            local_batch_size = user_embedding.shape[0]
            logits = logits * self.temperature_log.exp()
            labels = torch.arange(
                rank * local_batch_size, end=rank * local_batch_size + local_batch_size
            )

        else:
            logits = user_embedding @ job_embedding.T
            logits = logits * self.temperature_log.exp()
            labels = torch.arange(logits.shape[0], device=logits.device)

        loss = self.loss(logits, labels)
        return loss
