from dataclasses import dataclass
from typing import Literal, Union, Tuple, Any

# You’ll need to import these from wherever you defined them
# For example:
# from models.mlp import MLP
# from models.resnet import resnet
# from models.brazy import brazy_encoder, brazy_decoder

from models.fff_model import resnet, MLP, brazy_encoder, brazy_decoder

# define the dataclass for different architecture configurations
@dataclass
class MLPConfig:
    pixelwidth: int
    fff_dim: int
    fff_layers: int


@dataclass
class ResNetConfig:
    pixelwidth: int
    fff_dim: int
    fff_layers: int
    fff_resnet_blocks: int


@dataclass
class BrazyConfig:
    fff_c_small: int
    fff_f1_dim: int
    fff_f2_dim: int
    third_convolutional_layer: bool
    fff_batchnorm: bool
    fff_dropout: float


def get_encoder_and_decoder(
    fff_architecture: Literal["mlp", "resnet", "brazy"],
    config: Union[MLPConfig, ResNetConfig, BrazyConfig],
    device
) -> Tuple[Any, Any]:
    
    if fff_architecture == "mlp":
        assert isinstance(config, MLPConfig), "Config must be MLPConfig for 'mlp'"
        encoder = MLP(
            config.pixelwidth**2,
            config.fff_dim,
            config.pixelwidth**2,
            n_hidden_layers=config.fff_layers,
            device=device
        )
        decoder = MLP(
            config.pixelwidth**2,
            config.fff_dim,
            config.pixelwidth**2,
            n_hidden_layers=config.fff_layers,
            device=device
        )

    elif fff_architecture == "resnet":
        assert isinstance(config, ResNetConfig), "Config must be ResNetConfig for 'resnet'"
        encoder = resnet(
            input_dim=config.pixelwidth**2,
            hidden_dim=config.fff_dim,
            n_blocks=config.fff_resnet_blocks,
            output_dim=config.pixelwidth**2,
            hidden_layers=config.fff_layers
        )
        decoder = resnet(
            input_dim=config.pixelwidth**2,
            hidden_dim=config.fff_dim,
            n_blocks=config.fff_resnet_blocks,
            output_dim=config.pixelwidth**2,
            hidden_layers=config.fff_layers
        )

    elif fff_architecture == "brazy":
        assert isinstance(config, BrazyConfig), "Config must be BrazyConfig for 'brazy'"
        encoder = brazy_encoder(
            c_small=config.fff_c_small,
            f1_dim=config.fff_f1_dim,
            f2_dim=config.fff_f2_dim,
            third_conv=config.third_convolutional_layer,
            batchnorm=config.fff_batchnorm,
            p_dropout=config.fff_dropout
        )
        decoder = brazy_decoder(
            c_small=config.fff_c_small,
            f1_dim=config.fff_f1_dim,
            f2_dim=config.fff_f2_dim,
            third_conv=config.third_convolutional_layer,
            batchnorm=config.fff_batchnorm,
            p_dropout=config.fff_dropout
        )

    else:
        raise ValueError(f"Unknown architecture: {fff_architecture}")

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, decoder
