import torch
import torch.nn as nn
from collections import OrderedDict
from omegaconf import DictConfig
from pytorch_tabular.models.common.layers import AppendCLSToken, Embedding2dLayer, TransformerEncoderBlock
from pytorch_tabular.models.common.layers.batch_norm import BatchNorm1d
from pytorch_tabular.models.base_model import BaseModel


class HorseRaceAwareFTTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.hparams = config
        self._build_network()

    def _build_network(self):
        # Feature-level transformer blocks (within each horse)
        self.feature_transformer_blocks = OrderedDict()
        for i in range(self.hparams.feature_num_blocks):
            self.feature_transformer_blocks[f"feature_block_{i}"] = TransformerEncoderBlock(
                input_embed_dim=self.hparams.feature_embed_dim,
                num_heads=self.hparams.feature_num_heads,
                ff_hidden_multiplier=self.hparams.ff_hidden_multiplier,
                ff_activation=self.hparams.transformer_activation,
                attn_dropout=self.hparams.attn_dropout,
                ff_dropout=self.hparams.ff_dropout,
                add_norm_dropout=self.hparams.add_norm_dropout,
                keep_attn=self.hparams.attn_feature_importance,
            )
        self.feature_transformer_blocks = nn.Sequential(self.feature_transformer_blocks)

        # Project horse feature representation to horse embedding dimension
        self.horse_projection = nn.Linear(self.hparams.feature_embed_dim, self.hparams.horse_embed_dim)

        # Add race-level CLS token for aggregating horse information
        self.add_race_cls = AppendCLSToken(
            d_token=self.hparams.horse_embed_dim,
            initialization=self.hparams.embedding_initialization,
        )

        # Horse-level transformer blocks (across horses in race)
        self.horse_transformer_blocks = OrderedDict()
        for i in range(self.hparams.horse_num_blocks):
            self.horse_transformer_blocks[f"horse_block_{i}"] = TransformerEncoderBlock(
                input_embed_dim=self.hparams.horse_embed_dim,
                num_heads=self.hparams.horse_num_heads,
                ff_hidden_multiplier=self.hparams.ff_hidden_multiplier,
                ff_activation=self.hparams.transformer_activation,
                attn_dropout=self.hparams.attn_dropout,
                ff_dropout=self.hparams.ff_dropout,
                add_norm_dropout=self.hparams.add_norm_dropout,
                keep_attn=self.hparams.attn_feature_importance,
            )
        self.horse_transformer_blocks = nn.Sequential(self.horse_transformer_blocks)

        # Store attention weights for feature importance
        if self.hparams.attn_feature_importance:
            self.feature_attention_weights_ = [None] * self.hparams.feature_num_blocks
            self.horse_attention_weights_ = [None] * self.hparams.horse_num_blocks

        # Batch normalization for continuous inputs
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(self.hparams.continuous_dim, self.hparams.virtual_batch_size)

        self.output_dim = self.hparams.horse_embed_dim

    def _build_embedding_layer(self):
        return Embedding2dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.feature_embed_dim,  # Use feature embedding dim
            shared_embedding_strategy=getattr(self.hparams, "share_embedding_strategy", "fraction"),
            frac_shared_embed=getattr(self.hparams, "shared_embedding_fraction", 0.25),
            embedding_bias=self.hparams.embedding_bias,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            embedding_dropout=self.hparams.embedding_dropout,
            initialization=self.hparams.embedding_initialization,
            virtual_batch_size=getattr(self.hparams, "virtual_batch_size", None),
        )

    def forward_pass(self, batch):
        """
        Custom forward pass for race-grouped data.

        Args:
            batch: Dictionary containing:
                - 'continuous': (batch_size, max_horses, num_continuous)
                - 'categorical': (batch_size, max_horses, num_categorical)
                - 'target': (batch_size, max_horses)
                - 'attention_mask': (batch_size, max_horses)
        """
        continuous = batch["continuous"]  # (batch_size, max_horses, num_continuous)
        categorical = batch["categorical"]  # (batch_size, max_horses, num_categorical)
        attention_mask = batch["attention_mask"]  # (batch_size, max_horses)
        targets = batch["target"]  # (batch_size, max_horses)

        batch_size, max_horses, _ = continuous.shape

        # Reshape to process all horses through embedding layer
        # Flatten: (batch_size * max_horses, num_features)
        continuous_flat = continuous.view(-1, continuous.shape[-1])
        categorical_flat = categorical.view(-1, categorical.shape[-1])

        # Create input dict for embedding layer
        embed_input = {"continuous": continuous_flat, "categorical": categorical_flat}

        # Process through embedding layer
        embedded = self.embed_input(embed_input)  # (batch_size * max_horses, embed_dim)

        # Reshape back to race structure
        embed_dim = embedded.shape[-1]
        embedded_races = embedded.view(batch_size, max_horses, embed_dim)

        # Process through backbone with attention mask
        race_output = self.compute_backbone_with_mask(embedded_races, attention_mask)

        # Process through head to get predictions
        predictions = self.compute_head(race_output)  # (batch_size, max_horses, num_classes)

        return predictions, targets

    def compute_backbone_with_mask(self, x, attention_mask):
        """
        Process through backbone with attention masking for variable race sizes.

        Args:
            x: (batch_size, max_horses, embed_dim)
            attention_mask: (batch_size, max_horses) - 1 for real horses, 0 for padding
        """
        batch_size, max_horses, embed_dim = x.shape

        # Flatten for processing through your transformer
        x_flat = x.view(-1, embed_dim).unsqueeze(1)  # (batch_size * max_horses, 1, embed_dim)

        # Process through backbone
        backbone_output = self.backbone(x_flat)  # (batch_size * max_horses, output_dim)

        # Reshape back to race structure
        output_dim = backbone_output.shape[-1]
        race_output = backbone_output.view(batch_size, max_horses, output_dim)

        # Apply attention mask (zero out padded horses)
        attention_mask_expanded = attention_mask.unsqueeze(-1)  # (batch_size, max_horses, 1)
        race_output = race_output * attention_mask_expanded

        return race_output

    def _calculate_feature_importance(self):
        # Implement feature importance calculation combining both levels
        if not self.hparams.attn_feature_importance:
            return

        # This is a simplified version - you might want to combine
        # feature-level and horse-level attention weights
        if hasattr(self, "feature_attention_weights_") and self.feature_attention_weights_[0] is not None:
            n, h, f, _ = self.feature_attention_weights_[0].shape
            device = self.feature_attention_weights_[0].device
            L = len(self.feature_attention_weights_)

            self.local_feature_importance = torch.zeros((n, f), device=device)
            for attn_weights in self.feature_attention_weights_:
                self.local_feature_importance += attn_weights[:, :, :, -1].sum(dim=1)

            self.local_feature_importance = (1 / (h * L)) * self.local_feature_importance[:, :-1]
            self.feature_importance_ = self.local_feature_importance.mean(dim=0).detach().cpu().numpy()


class HorseRaceAwareModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head

    def _build_network(self):
        # Initialize your custom backbone
        self._backbone = HorseRaceAwareFTTransformer(self.hparams)

        # Initialize embedding layer
        self._embedding_layer = self._backbone._build_embedding_layer()

        # Initialize head (classification/regression layer)
        self._head = self._get_head_from_config()

    def feature_importance(self):
        """Extract feature importance from attention weights."""
        if self.hparams.attn_feature_importance:
            if hasattr(self._backbone, "_calculate_feature_importance"):
                self._backbone._calculate_feature_importance()
            return super().feature_importance()
        else:
            raise ValueError("If you want Feature Importance, `attn_feature_importance` should be `True`.")
