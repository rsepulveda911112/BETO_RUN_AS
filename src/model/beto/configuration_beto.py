""" RoBERTa configuration """
from transformers import BertConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BetoConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel`.
    It is used to instantiate an RoBERTa model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`.
    It reuses the same defaults. Please check the parent class for more information.

    Example::

        # >>> from transformers import SpanbertaConfig, SpanbertaModel
        #
        # >>> # Initializing a RoBERTa configuration
        # >>> configuration = RobertaConfig()
        #
        # >>> # Initializing a model from the configuration
        # >>> model = RobertaModel(configuration)
        #
        # >>> # Accessing the model configuration
        # >>> configuration = model.config
    """
    model_type = "beto"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs betoConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
