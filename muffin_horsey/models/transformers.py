from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FTTransformerConfig, NodeConfig, GANDALFConfig
from muffin_horsey.feature_processor import DataFrameInfo
from pytorch_tabular.models.stacking import StackingModelConfig
from pytorch_tabular import TabularModel
import warnings
from muffin_horsey.models.loss_fn import FocalLoss, calculate_optimal_focal_loss_params, get_class_imbalance_info
import polars as pl
import pandas
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, matthews_corrcoef
from pathlib import Path
import petname
from sklearn.dummy import DummyClassifier
import torch


def train_model(
    dataset_config: DataFrameInfo,
    train_set: pandas.DataFrame,
    validation_set: pandas.DataFrame,
    eval_set: pandas.DataFrame,
    live_player_df: pandas.DataFrame | None = None,
) -> tuple[pandas.DataFrame, pandas.DataFrame | None]:

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

    n_samples = len(train_set)
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

        # Loss function calculations.
        alpha, gamma = calculate_optimal_focal_loss_params(target_series=train_set[dataset_config.target_cols[0]])
        imbalance_info = get_class_imbalance_info(target_series=train_set[dataset_config.target_cols[0]])

        print(f"Class imbalance info: {imbalance_info}")
        print(f"Calculated focal loss params - alpha: {alpha:.3f}, gamma: {gamma:.1f}")

        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

        # Model fitting.
        tabular_model.fit(train=train_set, validation=validation_set, loss=loss_fn)

        # Run evaluation on test data.
        print("Evaluating model...")

        eval_df = tabular_model.predict(eval_set, include_input_features=True)

        predict_df = None

        if live_player_df is not None:
            print("Inferencing live player...")
            predict_df = tabular_model.predict(live_player_df, include_input_features=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return eval_df, predict_df


def run_eval(dataset_config: DataFrameInfo, live_player_request: pl.DataFrame | None) -> Path:

    # ===== Train the stacked models. =====

    train_set_pandas = dataset_config.train_set.to_pandas()
    validation_set_pandas = dataset_config.validation_set.to_pandas()
    eval_set_pandas = dataset_config.eval_set.to_pandas()
    live_player_request_pandas = live_player_request.to_pandas()

    eval_predictions_df, live_predictions_df = train_model(
        dataset_config=dataset_config,
        train_set=train_set_pandas,
        validation_set=validation_set_pandas,
        eval_set=eval_set_pandas,
        live_player_df=live_player_request_pandas,
    )

    # ===== Polars conversion. =====

    # Convert to polars for easier manipulation.
    eval_predictions_df = pl.from_pandas(eval_predictions_df)
    live_predictions_df = pl.from_pandas(live_predictions_df) if live_predictions_df is not None else None
    train_data_df = pl.from_pandas(train_set_pandas)
    test_data_df = pl.from_pandas(eval_set_pandas)

    # ===== Model evaluations. =====

    target_predict = f"{dataset_config.target_cols[0]}_prediction"
    target = dataset_config.target_cols[0]

    # Pull X/y as NumPy.
    x_train = train_data_df.drop(target).to_numpy()
    y_train = train_data_df[target].to_numpy()

    x_test = test_data_df.drop(target).to_numpy()
    y_true = test_data_df[target].to_numpy()

    y_pred = eval_predictions_df[target_predict].to_numpy()

    eval_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    report = classification_report(y_true=y_true, y_pred=y_pred)

    print(f"Accuracy score: {eval_accuracy:.2f}")
    print(f"{report}")

    # Cohen’s Kappa or MCC.
    kappa = cohen_kappa_score(y1=y_true, y2=y_pred)
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

    print(f"Cohen’s Kappa: {kappa:.3f}, MCC: {mcc:.3f}")

    # ===== Compare against a dummy “majority‐class” baseline. =====

    dummy = DummyClassifier(strategy="most_frequent")

    dummy.fit(X=x_train, y=y_train)
    y_dummy = dummy.predict(x_test)

    print(f"\n===== Dummy Baseline. =====\n")
    print(f"Dummy Accuracy score: {accuracy_score(y_true=y_true, y_pred=y_dummy)}")
    print(f"{classification_report(y_true=y_true, y_pred=y_dummy)}")

    # ===== Predictions on live data. =====

    print(f"\n===== Live data predictions. =====\n")
    if live_predictions_df is not None:
        print(live_predictions_df)
    else:
        print("No live data provided for prediction.")

    # ===== Save the model after training. =====

    curr_dir = Path.cwd()

    model_name = f"ftt_{petname.Generate()}"
    model_file_path = curr_dir.joinpath("checkpoints", model_name)

    # Disabled for debugging and testing.
    # logger.debug("Saving model...")
    # tabular_model.save_model(str(model_file_path))

    return model_file_path
