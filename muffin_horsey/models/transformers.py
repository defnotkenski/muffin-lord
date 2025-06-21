from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig, NodeConfig, GANDALFConfig
from muffin_horsey.feature_processor import DataFrameInfo
from pytorch_tabular.models.stacking import StackingModelConfig
from pytorch_tabular import TabularModel
import warnings
from muffin_horsey.models.loss_fn import FocalLoss, calculate_optimal_focal_loss_params, get_class_imbalance_info


def train_model(dataset_config: DataFrameInfo) -> None:

    data_config = DataConfig(
        target=dataset_config.target_cols,
        continuous_cols=dataset_config.continuous_cols,
        categorical_cols=dataset_config.categorical_cols,
        encode_date_columns=False,
        # date_columns=[("game_date", "D", "%Y-%m-%d")],
        # continuous_feature_transform="quantile_normal",
        # normalize_continuous_features=True,
    )

    trainer_config = TrainerConfig(
        auto_lr_find=False,
        batch_size=256,
        max_epochs=50,
        load_best=True,
        gradient_clip_val=1.0,
        early_stopping="valid_loss",
        early_stopping_mode="min",
        early_stopping_patience=5,
        checkpoints="valid_loss",
        checkpoints_mode="min",
    )

    n_samples = len(dataset_config.train_set)
    batch_size = trainer_config.batch_size
    steps_per_epoch = (n_samples + batch_size - 1) // batch_size

    optimizer_config = OptimizerConfig(
        optimizer="AdamW",
        optimizer_params={"weight_decay": 0.01},
        lr_scheduler="OneCycleLR",
        lr_scheduler_interval="step",
        lr_scheduler_params={
            "max_lr": 1e-4,  # Prev. 3e-4
            "steps_per_epoch": steps_per_epoch,
            "epochs": trainer_config.max_epochs,
        },
    )

    # ===== Model configuration and setup. =====

    task = "classification"

    model_config_ft = FTTransformerConfig(
        task=task,
        input_embed_dim=128,  # Increased from 64.
        num_heads=8,
        num_attn_blocks=6,  # Increased from 4.
        ff_dropout=0.15,  # Increased from 0.1
        attn_dropout=0.1,
    )

    model_config_node = NodeConfig(
        task=task,
    )

    model_config_gandalf = GANDALFConfig(
        task=task,
        batch_norm_continuous_input=True,
    )

    stacking_model = StackingModelConfig(
        task=task,
        model_configs=[model_config_ft, model_config_gandalf, model_config_node],
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=stacking_model,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
    )

    # ===== Train the model. =====

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        train_set_pandas = dataset_config.train_set.to_pandas()
        validation_set_pandas = dataset_config.validation_set.to_pandas()
        eval_set_pandas = dataset_config.eval_set.to_pandas()

        # Loss function calculations.
        alpha, gamma = calculate_optimal_focal_loss_params(
            target_series=train_set_pandas[dataset_config.target_cols[0]]
        )
        imbalance_info = get_class_imbalance_info(target_series=train_set_pandas[dataset_config.target_cols[0]])

        print(f"Class imbalance info: {imbalance_info}")
        print(f"Calculated focal loss params - alpha: {alpha:.3f}, gamma: {gamma:.1f}")

        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

        # Model fitting.
        tabular_model.fit(train=train_set_pandas, validation=validation_set_pandas, loss=loss_fn)

        # Run evaluation on test data.
        print("Evaluating model...")

        eval_df = tabular_model.predict(eval_set_pandas, include_input_features=True)
        print(eval_df)

    return
