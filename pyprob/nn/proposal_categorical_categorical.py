import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Categorical


class ProposalCategoricalCategorical(nn.Module):
    def __init__(self, input_shape, num_categories, num_layers=2, hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([num_categories]),
                                        num_layers=num_layers,
                                        activation=torch.relu,
                                        activation_last=None,
                                        hidden_dim=hidden_dim)
        self._total_train_iterations = 0
        self._logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        logits = self._logsoftmax(x).view(batch_size, -1)
        return Categorical(logits=logits)
