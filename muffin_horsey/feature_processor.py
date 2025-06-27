from typing import Literal
import polars
import polars as pl
from muffin_horsey.feature_generator import generate_train_features
from typing import NamedTuple
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
from schema import COLUMN_TYPES


class DataFrameInfo(NamedTuple):
    df: pl.DataFrame
    train_set: pl.DataFrame
    validation_set: pl.DataFrame
    eval_set: pl.DataFrame
    continuous_cols: list[str]
    categorical_cols: list[str]
    target_cols: list[str]


class FeatureProcessor:
    def __init__(self, df: pl.DataFrame, target_type: Literal["win", "show", "place"]):
        self.base_df: pl.DataFrame = df

        self.processed_df: pl.DataFrame | None = None
        self.features_raw_df: pl.DataFrame | None = None

        self.target_type: str = target_type

        all_features = generate_train_features(lag_count=1, other_count=4)
        self.train_features = all_features

    @staticmethod
    def _build_prediction_safe_features(working_df: pl.DataFrame) -> pl.DataFrame:
        """Features that can be calculated at prediction time"""

        # ===== Add distance_furlongs column. =====

        working_df = working_df.with_columns((pl.col("distance").cast(pl.Float64) / 100).alias("distance_furlongs"))

        # ===== Add field_size column. =====

        working_df = working_df.with_columns(
            pl.count().over(["race_date", "track_code", "race_number"]).alias("field_size")
        )

        # ===== Add rank_in_odds column. =====

        working_df = working_df.with_columns(
            pl.col("dollar_odds")
            .rank(method="ordinal")
            .over(["race_date", "track_code", "race_number"])
            .alias("rank_in_odds")
        )

        # ===== Add days_since_last_race column. =====

        working_df = working_df.with_columns(
            (pl.col("race_date") - pl.col("last_pp_race_date")).dt.total_days().alias("days_since_last_race")
        )

        # ===== Add trainer_full_name. =====

        working_df = working_df.with_columns(
            pl.concat_str([pl.col("trainer_first_name"), pl.col("trainer_last_name")], separator=" ").alias(
                "trainer_full_name"
            )
        )

        return working_df

    @staticmethod
    def _process_lag_races(working_df: pl.DataFrame, lookup_df: pl.DataFrame | None = None) -> pl.DataFrame:
        # Use lookup_df if provided, otherwise use feature_df for self-join
        source_df = lookup_df if lookup_df is not None else working_df

        original_cols = working_df.columns

        source_df_select = [
            # Start of join cols.
            "race_date",
            "race_number",
            "track_code",
            "horse_name",
            # Start of cols to add with suffix.
            "race_type",
            "distance_furlongs",
            "race_purse",
            "field_size",
            "course_surface",
            "class_rating",
            "track_conditions",
            "runup_distance",
            "rail_distance",
            "sealed",
            "rank_in_odds",
            "dollar_odds",
            "days_since_last_race",
            "trainer_win_pct",
            "start_position",
            "point_of_call_1_position",
            "point_of_call_1_lengths",
            "point_of_call_5_position",
            "point_of_call_5_lengths",
            "point_of_call_final_position",
            "point_of_call_final_lengths",
            "speed_rating",
            "race_speed_vs_par",
            "horse_speed_vs_par",
            "horse_time_vs_winner",
            "speed_rating_vs_field_avg",
            "speed_rating_vs_winner",
        ]

        working_df = working_df.join(
            source_df.select(source_df_select),
            left_on=["last_pp_race_date", "last_pp_track_code", "last_pp_race_number", "horse_name"],
            right_on=["race_date", "track_code", "race_number", "horse_name"],
            how="left",
            suffix="_recent_0",
            coalesce=False,
        )

        # ===== Rename lag columns with approprate prefix. =====

        new_cols = [col for col in working_df.columns if col not in original_cols]

        working_df = working_df.rename(
            {col_name: f"{col_name}_recent_0" for col_name in new_cols if col_name in source_df_select}
        )

        working_df = working_df.rename(
            {col: f"recent_0_{col.replace("_recent_0", "")}" for col in working_df.columns if col.endswith("_recent_0")}
        )

        return working_df

    @staticmethod
    def _process_opponents(working_df: pl.DataFrame, lookup_df: pl.DataFrame | None = None) -> pl.DataFrame:
        # Use lookup_df if provided, otherwise use working_df for self-join
        source_df = lookup_df if lookup_df is not None else working_df

        # Generate the current race features for the top 4 opponent horses. (ranked by dollar_odds)
        opp_cols_to_add = [
            "horse_name",
            "dollar_odds",
            "rank_in_odds",
            "trainer_win_pct",
            "days_since_last_race",
            "last_pp_race_date",
            "last_pp_track_code",
            "last_pp_race_number",
        ]

        race_data = (
            working_df.group_by(["race_date", "track_code", "race_number"])
            .agg([pl.col(col).sort_by("rank_in_odds").alias(f"all_{col}") for col in opp_cols_to_add])
            .sort(["race_date", "track_code", "race_number"])
        )

        working_df = working_df.join(race_data, on=["race_date", "track_code", "race_number"])

        # Create offset indices in order to get true opponents without current horse.
        working_df = working_df.with_columns(
            pl.when(pl.col("rank_in_odds") == 1)
            .then(pl.lit([1, 2, 3, 4]))
            .when(pl.col("rank_in_odds") == 2)
            .then(pl.lit([0, 2, 3, 4]))
            .when(pl.col("rank_in_odds") == 3)
            .then(pl.lit([0, 1, 3, 4]))
            .when(pl.col("rank_in_odds") == 4)
            .then(pl.lit([0, 1, 2, 4]))
            .otherwise(pl.lit([0, 1, 2, 3]))
            .alias("opponent_indices")
        )

        # ===== Use the offset opponent indices to grab the appropriate values for each col. =====

        for col in opp_cols_to_add:
            working_df = working_df.with_columns(
                [
                    pl.col(f"all_{col}")
                    .list.get(pl.col("opponent_indices").list.get(idx), null_on_oob=True)
                    .alias(f"opp_{idx + 1}_{col}")
                    for idx in range(4)
                ]
            )

        # ===== Create lag cols for the opponents. =====

        for i in range(4):
            working_df = working_df.join(
                source_df.select(
                    [
                        "race_date",
                        "track_code",
                        "race_number",
                        "horse_name",
                        "course_surface",
                        "distance_furlongs",
                        "class_rating",
                        "dollar_odds",
                        "trainer_win_pct",
                        "start_position",
                        "official_final_position",
                        "speed_rating",
                        "race_speed_vs_par",
                        "horse_speed_vs_par",
                        "speed_rating_vs_field_avg",
                        "speed_rating_vs_winner",
                    ]
                ),
                left_on=[
                    f"opp_{i+1}_last_pp_race_date",
                    f"opp_{i+1}_last_pp_track_code",
                    f"opp_{i+1}_last_pp_race_number",
                    f"opp_{i+1}_horse_name",
                ],
                right_on=["race_date", "track_code", "race_number", "horse_name"],
                how="left",
                suffix=f"_opp_{i+1}_recent_0",
            )

            # Rename cols with the appropriate prefix.
            working_df = working_df.rename(
                {
                    col: f"opp_{i+1}_recent_0_{col.replace(f"_opp_{i+1}_recent_0", "")}"
                    for col in working_df.columns
                    if col.endswith(f"_opp_{i+1}_recent_0")
                }
            )

        return working_df

    def _build_features(self) -> bool:
        # ===== Set the base or working dataframe. =====

        feature_df = self.base_df

        # ===== Create new features based on existing columns. =====

        feature_df = self._build_prediction_safe_features(working_df=feature_df)

        # ===== Pipeline to add trainer_win_pct column. =====

        feature_df = feature_df.with_columns(
            (
                pl.col("official_final_position")
                .is_in([1, 2, 3])
                .cum_sum()
                .over("trainer_full_name", order_by=["race_date", "track_code", "race_number"])
                .cast(pl.Int64)
                - 1
            )
            .clip(lower_bound=0)
            .alias("trainer_wins")
        )

        feature_df = feature_df.with_columns(
            pl.int_range(pl.len())
            .over("trainer_full_name", order_by=["race_date", "track_code", "race_number"])
            .cast(pl.Int64)
            .clip(lower_bound=0)
            .alias("trainer_entries")
        )

        feature_df = feature_df.with_columns(
            pl.when(pl.col("trainer_entries") == 0)
            .then(0)
            .otherwise(pl.col("trainer_wins") / pl.col("trainer_entries"))
            .alias("trainer_win_pct")
        )

        # ===== Results-derived feature calculations. =====

        # Calculate race_speed_vs_par.
        feature_df = feature_df.with_columns(
            pl.when(pl.col("par_time") != 0.00)
            .then(pl.col("win_time") - pl.col("par_time"))
            .otherwise(pl.lit(None))
            .alias("race_speed_vs_par")
        )

        # Calculate horse_speed_vs_par.
        length_seconds = 0.2

        feature_df = feature_df.with_columns(
            (pl.col("point_of_call_final_lengths") * length_seconds + pl.col("win_time")).alias("horse_finish_time")
        )

        feature_df = feature_df.with_columns(
            (pl.col("horse_finish_time") - pl.col("par_time")).alias("horse_speed_vs_par")
        )

        # Calculate the horse_time_vs_winner.
        feature_df = feature_df.with_columns(
            (pl.col("horse_finish_time") - pl.col("win_time")).alias("horse_time_vs_winner")
        )

        # Calculate the speed_rating_vs_field_avg.
        feature_df = feature_df.with_columns(
            ((pl.col("speed_rating").sum() - pl.col("speed_rating")) / (pl.len() - 1))
            .over(["race_date", "track_code", "race_number"])
            .alias("field_avg_speed_rating")
        )

        feature_df = feature_df.with_columns(
            (pl.col("speed_rating") - pl.col("field_avg_speed_rating")).alias("speed_rating_vs_field_avg")
        )

        # Calculate the speed_rating_vs_winner.
        feature_df = feature_df.with_columns(
            pl.col("speed_rating")
            .get(pl.col("official_final_position").arg_min())
            .over(["race_date", "track_code", "race_number"])
            .alias("speed_rating_winner")
        )

        feature_df = feature_df.with_columns(
            (pl.col("speed_rating") - pl.col("speed_rating_winner")).alias("speed_rating_vs_winner")
        )

        # ===== Create lag races for horses. =====

        feature_df = self._process_lag_races(working_df=feature_df)

        # ===== Process opponent horses. =====

        feature_df = self._process_opponents(working_df=feature_df)

        # ===== Generate target columns. =====

        if self.target_type == "win":
            feature_df = feature_df.with_columns(
                (pl.col("official_final_position") == 1).cast(pl.Int64).alias("target")
            )
        elif self.target_type == "show":
            feature_df = feature_df.with_columns(
                (pl.col("official_final_position") <= 2).cast(pl.Int64).alias("target")
            )
        else:
            feature_df = feature_df.with_columns(
                (pl.col("official_final_position") <= 3).cast(pl.Int64).alias("target")
            )

        # ===== Final sort for redundancy. =====

        feature_df = feature_df.sort(["race_date", "track_code", "race_number"])

        # ===== Set the raw features df to be used as a helper df when making predictions downstream. =====
        self.features_raw_df = feature_df

        # ===== Select only training columns + some helper columns for downstream logic. =====

        feature_df = feature_df.select(["race_date", "race_number", *self.train_features])
        self.processed_df = feature_df

        return True

    def _handle_missing_values(self, working_df: pl.DataFrame | None = None) -> pl.DataFrame | None:
        base_df: pl.DataFrame = working_df if working_df is not None else self.processed_df

        # Add indicator columns for cols susceptible to missing data.
        base_df = base_df.with_columns(
            pl.all().exclude(["race_date", "race_number", "target"]).is_null().cast(pl.Int64).name.suffix("_is_null")
        )

        # Fill nulls with a sentinel value like -999. Do not go bigger in order to prevent gradient issues.
        base_df = base_df.with_columns(pl.col(pl.selectors.NUMERIC_DTYPES).fill_null(-999))

        # After making changes.
        _null_count_after = base_df.null_count()

        # Final sort for redundancy.
        base_df = base_df.sort(["race_date", "track_code", "race_number"])

        # Only update self.processed_df if we're working on the training data.
        if working_df is None:
            self.processed_df = base_df
            return None

        return base_df

    def get_dataframe(self) -> DataFrameInfo:
        """This function serves as the orchestrator of various methods in order to output a train-ready dataframe."""

        # Extract features from data.
        self._build_features()

        # Clean up data to prepare for transformer. Requires pre-selected columns, do not give it the full columns.
        self._handle_missing_values()

        # No more mutations of self.processed_df beyond this point.
        working_df = self.processed_df

        # Organize into categorical, continuous, and target cols for model.
        continuous_cols = working_df.select(
            pl.selectors.numeric().exclude(["race_date", "race_number", "target"])
        ).columns
        string_cols = working_df.select(pl.selectors.string()).columns
        target_cols = ["target"]

        # Allocation of the processed dataset for train, validation, and evaluation.
        # Get unique races.
        unique_races = working_df.select(["race_date", "track_code", "race_number"]).unique(maintain_order=True)

        # Split race identifiers.
        race_train, race_temp = train_test_split(unique_races, random_state=42, shuffle=False, test_size=0.10)
        race_validation, race_eval = train_test_split(race_temp, random_state=42, shuffle=False, test_size=0.50)

        # Filter original dataset by race splits.
        train_set = working_df.join(race_train, on=["race_date", "track_code", "race_number"])
        validation_set = working_df.join(race_validation, on=["race_date", "track_code", "race_number"])
        eval_set = working_df.join(race_eval, on=["race_date", "track_code", "race_number"])

        return DataFrameInfo(
            df=working_df,
            train_set=train_set,
            validation_set=validation_set,
            eval_set=eval_set,
            continuous_cols=continuous_cols,
            categorical_cols=string_cols,
            target_cols=target_cols,
        )

    def get_predict_dataframe(self) -> pl.DataFrame:
        path_to_predict_yaml = Path.cwd() / "predict.yaml"

        with open(path_to_predict_yaml, "r") as r_yaml:
            predict_data = yaml.safe_load(r_yaml)

        base_df = polars.from_dict({**predict_data["current_race"], **predict_data["current_horse"]})

        base_df = base_df.cast({col: dtype for col, dtype in COLUMN_TYPES.items() if col in base_df.columns})

        base_df = base_df.with_columns(polars.col("last_pp_race_date").str.to_datetime())
        base_df = base_df.cast({"race_date": pl.Datetime})

        base_df = self._build_prediction_safe_features(working_df=base_df)

        # ===== Add trainer_win_pct column. =====

        historical_df = self.features_raw_df

        trainer_stats = (
            historical_df.filter(pl.col("race_date") < base_df["race_date"][0])
            .group_by("trainer_full_name")
            .agg([(pl.col("official_final_position").is_in([1, 2, 3]).sum() / pl.len()).alias("trainer_win_pct")])
        )

        base_df = base_df.join(trainer_stats, on="trainer_full_name", how="left")
        base_df = base_df.with_columns(pl.col("trainer_win_pct").fill_null(0))

        # Process lag races for the prediction df.
        base_df = self._process_lag_races(working_df=base_df, lookup_df=historical_df)

        # Process opponent columns.
        base_df = self._process_opponents(working_df=base_df, lookup_df=historical_df)

        # Process indicator columns.
        base_df = self._handle_missing_values(working_df=base_df)

        # Verify that the training df and the prediction df structure are identical.
        train_df_cols = self.processed_df.columns
        predict_df_cols = base_df.columns

        _col_difference = [col for col in train_df_cols if col not in predict_df_cols]

        return base_df
