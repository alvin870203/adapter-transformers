import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


logger = logging.getLogger(__name__)

class SwinBlockAdaptersMixin:
    """Adds adapters to the SwinBlock module of Swin."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class SwinModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the SwinModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        idx = 0
        for stage in self.encoder.layers:
            for layer in stage.blocks:
                yield idx, layer
                idx += 1
