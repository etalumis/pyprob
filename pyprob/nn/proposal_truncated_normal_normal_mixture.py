import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Normal, TruncatedNormal, Mixture


class ProposalTruncatedNormalNormalMixture(nn.Module):
    def __init__(self, input_shape, num_layers=2, mixture_components=10):
        super().__init__()
        # Currently only supports event_shape=torch.Size([]) for the mixture components
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2*self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2*self._mixture_components:].view(batch_size, -1)
        stddevs = torch.exp(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(batch_size, -1)
        prior_stddevs = torch.stack([v.distribution.stddev for v in prior_variables]).view(batch_size, -1)
        prior_means = prior_means.expand_as(means)
        prior_stddevs = prior_stddevs.expand_as(stddevs)
        prior_high = torch.stack([v.distribution.high for v in prior_variables]).view(batch_size, -1)
        prior_low = torch.stack([v.distribution.low for v in prior_variables]).view(batch_size, -1)
        high = prior_high.expand_as(means)
        low = prior_low.expand_as(means)
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        distributions = [TruncatedNormal(means[:, i:i+1].view(batch_size), stddevs[:, i:i+1].view(batch_size), low[:, i:i+1].view(batch_size), high[:, i:i+1].view(batch_size)) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)
