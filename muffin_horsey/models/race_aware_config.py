from dataclasses import dataclass, field
from pytorch_tabular.config import ModelConfig


@dataclass
class HorseRaceFTTransformerConfig(ModelConfig):
    """Configuration for Horse Race FTTransformer with cross-horse attention."""

    # Horse race specific parameters
    max_horses_per_race: int = field(default=16, metadata={"help": "Maximum number of horses per race"})

    # Feature-level transformer parameters (within each horse)
    feature_embed_dim: int = field(default=64, metadata={"help": "Embedding dimension for horse features"})
    feature_num_heads: int = field(default=8, metadata={"help": "Number of attention heads for feature attention"})
    feature_num_blocks: int = field(default=3, metadata={"help": "Number of feature-level transformer blocks"})

    # Horse-level transformer parameters (across horses in race)
    horse_embed_dim: int = field(default=128, metadata={"help": "Embedding dimension for horse representations"})
    horse_num_heads: int = field(default=8, metadata={"help": "Number of attention heads for horse attention"})
    horse_num_blocks: int = field(default=4, metadata={"help": "Number of horse-level transformer blocks"})

    # Standard transformer parameters
    ff_hidden_multiplier: int = field(default=4, metadata={"help": "Feed-forward hidden layer multiplier"})
    transformer_activation: str = field(
        default="GEGLU", metadata={"help": "Activation function (GEGLU, ReGLU, SwiGLU)"}
    )

    # Dropout parameters
    attn_dropout: float = field(default=0.1, metadata={"help": "Attention dropout rate"})
    ff_dropout: float = field(default=0.1, metadata={"help": "Feed-forward dropout rate"})
    add_norm_dropout: float = field(default=0.1, metadata={"help": "Add & norm dropout rate"})
    embedding_dropout: float = field(default=0.0, metadata={"help": "Embedding dropout rate"})

    # Embedding parameters
    embedding_initialization: str = field(
        default="kaiming_uniform", metadata={"help": "Embedding initialization method"}
    )
    embedding_bias: bool = field(default=True, metadata={"help": "Use bias in embedding layers"})

    # Feature importance
    attn_feature_importance: bool = field(
        default=True, metadata={"help": "Calculate attention-based feature importance"}
    )

    # Aggregation strategy for race prediction
    race_aggregation: str = field(
        default="cls_token", metadata={"help": "How to aggregate horse representations (cls_token, mean, max)"}
    )

    # Required PyTorch-Tabular metadata
    _module_src: str = field(default="muffin_horsey.models.horse_race_ft_transformer")
    _model_name: str = field(default="HorseRaceFTTransformerModel")
    _backbone_name: str = field(default="HorseRaceFTTransformerBackbone")
    _config_name: str = field(default="HorseRaceFTTransformerConfig")
