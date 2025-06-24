from typing import Literal
import polars
import polars as pl
from muffin_horsey.feature_generator import generate_train_features
from typing import NamedTuple
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml


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

        self.target_type: str = target_type

        all_features = generate_train_features(lag_count=1, other_count=4)
        self.train_features = all_features

    @staticmethod
    def _process_lag_races(feature_df: pl.DataFrame) -> pl.DataFrame:
        # Find horse's last race based on track code and race number.
        # feature_df = feature_df.with_columns(pl.col("last_pp_track_code").alias("track_code_recent_0"))

        feature_df = feature_df.join(
            feature_df.select(
                [
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
                    "par_time",
                    "win_time",
                ]
            ),
            left_on=["last_pp_race_date", "last_pp_track_code", "last_pp_race_number", "horse_name"],
            right_on=["race_date", "track_code", "race_number", "horse_name"],
            how="left",
            suffix="_recent_0",
        )

        # Rename lag columns with approprate prefix.
        feature_df = feature_df.rename(
            {col: f"recent_0_{col.replace("_recent_0", "")}" for col in feature_df.columns if col.endswith("_recent_0")}
        )

        # Calculate race_speed_vs_par.
        # Final time of winner - par time (in seconds or lengths).
        feature_df = feature_df.with_columns(
            pl.when(pl.col("recent_0_par_time") != 0.00)
            .then(pl.col("recent_0_win_time") - pl.col("recent_0_par_time"))
            .otherwise(pl.lit(None))
            .alias("recent_0_race_speed_vs_par")
        )

        # ===== Calculate horse_speed_vs_par. =====

        length_seconds = 0.2

        feature_df = feature_df.with_columns(
            (pl.col("recent_0_point_of_call_final_lengths") * length_seconds + pl.col("recent_0_win_time")).alias(
                "recent_0_horse_finish_time"
            )
        )

        feature_df = feature_df.with_columns(
            (pl.col("recent_0_horse_finish_time") - pl.col("recent_0_par_time")).alias("recent_0_horse_speed_vs_par")
        )

        # Calculate the horse_time_vs_winner.
        feature_df = feature_df.with_columns(
            (pl.col("recent_0_horse_finish_time") - pl.col("recent_0_win_time")).alias("recent_0_horse_time_vs_winner")
        )

        # ===== Calculate the recent_0_speed_rating_vs_field_avg. =====
        # Horse_rating - avg_field_rating.

        # First, create field averages for all races in the dataset
        race_field_stats = feature_df.group_by(["race_date", "track_code", "race_number"]).agg(
            [pl.col("speed_rating").alias("all_speed_ratings_in_race")]
        )

        # Then join this back to get the field data for each horse's recent_0 race
        feature_df = feature_df.join(
            race_field_stats,
            left_on=["last_pp_race_date", "last_pp_track_code", "last_pp_race_number"],
            right_on=["race_date", "track_code", "race_number"],
            how="left",
            suffix="_field_data",
        )

        # Now calculate field average excluding the current horse's speed rating
        feature_df = feature_df.with_columns(
            (
                (pl.col("all_speed_ratings_in_race").list.sum() - pl.col("recent_0_speed_rating"))
                / (pl.col("all_speed_ratings_in_race").list.len() - 1)
            ).alias("recent_0_field_avg_speed_rating")
        )

        # Horse's speed rating minus average of all other horses in that historical race
        feature_df = feature_df.with_columns(
            (pl.col("recent_0_speed_rating") - pl.col("recent_0_field_avg_speed_rating")).alias(
                "recent_0_speed_rating_vs_field_avg"
            )
        )

        # ===== Calculate the speed_rating_vs_winner. =====

        # Get winner's speed rating for each historical race
        race_winner_stats = feature_df.group_by(["race_date", "track_code", "race_number"]).agg(
            [pl.col("speed_rating").filter(pl.col("official_final_position") == 1).first().alias("winner_speed_rating")]
        )

        # Join to get winner's speed rating from each horse's historical race
        feature_df = feature_df.join(
            race_winner_stats,
            left_on=["last_pp_race_date", "last_pp_track_code", "last_pp_race_number"],
            right_on=["race_date", "track_code", "race_number"],
            how="left",
            suffix="_recent_0",
        )

        # Now you have recent_0_winner_speed_rating available
        feature_df = feature_df.rename({"winner_speed_rating": "recent_0_winner_speed_rating"})

        feature_df = feature_df.with_columns(
            (pl.col("recent_0_speed_rating") - pl.col("recent_0_winner_speed_rating")).alias(
                "recent_0_speed_rating_vs_winner"
            )
        )

        return feature_df

    @staticmethod
    def _process_opponents(base_df: pl.DataFrame) -> pl.DataFrame:
        # Generate the current race features for the top 4 opponent horses. (ranked by dollar_odds)
        race_data = (
            base_df.group_by(["race_date", "track_code", "race_number"])
            .agg(
                [
                    pl.col("horse_name").sort_by("rank_in_odds").alias("all_horse_name"),
                    pl.col("dollar_odds").sort_by("rank_in_odds").alias("all_dollar_odds"),
                    pl.col("rank_in_odds").sort_by("rank_in_odds").alias("all_rank_in_odds"),
                    pl.col("trainer_win_pct").sort_by("rank_in_odds").alias("all_trainer_win_pct"),
                    pl.col("days_since_last_race").sort_by("rank_in_odds").alias("all_days_since_last_race"),
                    pl.col("last_pp_race_date").sort_by("rank_in_odds").alias("all_last_pp_race_date"),
                    pl.col("last_pp_track_code").sort_by("rank_in_odds").alias("all_last_pp_track_code"),
                    pl.col("last_pp_race_number").sort_by("rank_in_odds").alias("all_last_pp_race_number"),
                ]
            )
            .sort(["race_date", "track_code", "race_number"])
        )

        base_df = base_df.join(race_data, on=["race_date", "track_code", "race_number"])

        # Create offset indices in order to get true opponents without current horse.
        base_df = base_df.with_columns(
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

        # Use the offset opponent indices to grab the appropriate values for each col.
        # Create opp_X_dollar_odds.
        base_df = base_df.with_columns(
            [
                pl.col("all_dollar_odds")
                .list.get(pl.col("opponent_indices").list.get(0), null_on_oob=True)
                .alias("opp_1_dollar_odds"),
                pl.col("all_dollar_odds")
                .list.get(pl.col("opponent_indices").list.get(1), null_on_oob=True)
                .alias("opp_2_dollar_odds"),
                pl.col("all_dollar_odds")
                .list.get(pl.col("opponent_indices").list.get(2), null_on_oob=True)
                .alias("opp_3_dollar_odds"),
                pl.col("all_dollar_odds")
                .list.get(pl.col("opponent_indices").list.get(3), null_on_oob=True)
                .alias("opp_4_dollar_odds"),
            ]
        )

        # Create opp_X_rank_in_odds.
        base_df = base_df.with_columns(
            [
                pl.col("all_rank_in_odds")
                .list.get(pl.col("opponent_indices").list.get(0), null_on_oob=True)
                .alias("opp_1_rank_in_odds"),
                pl.col("all_rank_in_odds")
                .list.get(pl.col("opponent_indices").list.get(1), null_on_oob=True)
                .alias("opp_2_rank_in_odds"),
                pl.col("all_rank_in_odds")
                .list.get(pl.col("opponent_indices").list.get(2), null_on_oob=True)
                .alias("opp_3_rank_in_odds"),
                pl.col("all_rank_in_odds")
                .list.get(pl.col("opponent_indices").list.get(3), null_on_oob=True)
                .alias("opp_4_rank_in_odds"),
            ]
        )

        # Create opp_X_days_since_last_race.
        base_df = base_df.with_columns(
            [
                pl.col("all_days_since_last_race")
                .list.get(pl.col("opponent_indices").list.get(0), null_on_oob=True)
                .alias("opp_1_days_since_last_race"),
                pl.col("all_days_since_last_race")
                .list.get(pl.col("opponent_indices").list.get(1), null_on_oob=True)
                .alias("opp_2_days_since_last_race"),
                pl.col("all_days_since_last_race")
                .list.get(pl.col("opponent_indices").list.get(2), null_on_oob=True)
                .alias("opp_3_days_since_last_race"),
                pl.col("all_days_since_last_race")
                .list.get(pl.col("opponent_indices").list.get(3), null_on_oob=True)
                .alias("opp_4_days_since_last_race"),
            ]
        )

        # Create opp_X_trainer_win_pct.
        base_df = base_df.with_columns(
            [
                pl.col("all_trainer_win_pct")
                .list.get(pl.col("opponent_indices").list.get(0), null_on_oob=True)
                .alias("opp_1_trainer_win_pct"),
                pl.col("all_trainer_win_pct")
                .list.get(pl.col("opponent_indices").list.get(1), null_on_oob=True)
                .alias("opp_2_trainer_win_pct"),
                pl.col("all_trainer_win_pct")
                .list.get(pl.col("opponent_indices").list.get(2), null_on_oob=True)
                .alias("opp_3_trainer_win_pct"),
                pl.col("all_trainer_win_pct")
                .list.get(pl.col("opponent_indices").list.get(3), null_on_oob=True)
                .alias("opp_4_trainer_win_pct"),
            ]
        )

        # Create opp_X_horse_name.
        base_df = base_df.with_columns(
            [
                pl.col("all_horse_name")
                .list.get(pl.col("opponent_indices").list.get(idx), null_on_oob=True)
                .alias(f"opp_{idx + 1}_horse_name")
                for idx in range(4)
            ]
        )

        # Create opp_X_last_pp_race_date.
        base_df = base_df.with_columns(
            [
                pl.col("all_last_pp_race_date")
                .list.get(pl.col("opponent_indices").list.get(idx), null_on_oob=True)
                .alias(f"opp_{idx+1}_last_pp_race_date")
                for idx in range(4)
            ]
        )

        # Create opp_X_last_pp_track_code.
        base_df = base_df.with_columns(
            [
                pl.col("all_last_pp_track_code")
                .list.get(pl.col("opponent_indices").list.get(idx), null_on_oob=True)
                .alias(f"opp_{idx + 1}_last_pp_track_code")
                for idx in range(4)
            ]
        )

        # Create opp_X_last_pp_race_number.
        base_df = base_df.with_columns(
            [
                pl.col("all_last_pp_race_number")
                .list.get(pl.col("opponent_indices").list.get(idx), null_on_oob=True)
                .alias(f"opp_{idx + 1}_last_pp_race_number")
                for idx in range(4)
            ]
        )

        # Join opponent's last race data.
        for i in range(4):
            base_df = base_df.join(
                base_df.select(
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
                        "race_speed_vs_par",
                        "horse_speed_vs_par",
                        "speed_rating",
                        "speed_rating_vs_field",
                        "speed_rating_vs_winner",
                        "start_position",
                        "official_final_position",
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
            base_df = base_df.rename(
                {
                    col: f"opp_{i+1}_recent_0_{col.replace(f"_opp_{i+1}_recent_0", "")}"
                    for col in base_df.columns
                    if col.endswith(f"_opp_{i+1}_recent_0")
                }
            )

        return base_df

    def _build_features(self, predict_df: pl.DataFrame | None = None) -> bool | pl.DataFrame:
        # Set the base or working dataframe.
        if predict_df is None:
            feature_df = self.base_df
        else:
            feature_df = predict_df

        # Create new features based on existing columns.
        # Add distance_furlongs column.
        feature_df = feature_df.with_columns((pl.col("distance").cast(pl.Float64) / 100).alias("distance_furlongs"))

        # Add field_size column.
        feature_df = feature_df.with_columns(
            pl.count().over(["race_date", "track_code", "race_number"]).alias("field_size")
        )

        # Add rank_in_odds column.
        feature_df = feature_df.with_columns(
            pl.col("dollar_odds")
            .rank(method="ordinal")
            .over(["race_date", "track_code", "race_number"])
            .alias("rank_in_odds")
        )

        # Add days_since_last_race column.
        feature_df = feature_df.with_columns(
            (pl.col("race_date") - pl.col("last_pp_race_date")).dt.total_days().alias("days_since_last_race")
        )

        # Pipeline to add trainer_win_pct_30d column.
        feature_df = feature_df.with_columns(
            pl.concat_str([pl.col("trainer_first_name"), pl.col("trainer_last_name")], separator=" ").alias(
                "trainer_full_name"
            )
        )

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

        # Results-derived feature calculations.
        # # Calculate race_speed_vs_par.
        # feature_df = feature_df.with_columns(
        #     pl.when(pl.col("par_time") != 0.00)
        #     .then(pl.col("win_time") - pl.col("par_time"))
        #     .otherwise(pl.lit(None))
        #     .alias("race_speed_vs_par")
        # )
        #
        # # Calculate horse_speed_vs_par.
        # length_seconds = 0.2
        #
        # feature_df = feature_df.with_columns(
        #     (pl.col("point_of_call_final_lengths") * length_seconds + pl.col("win_time")).alias("horse_finish_time")
        # )
        #
        # feature_df = feature_df.with_columns(
        #     (pl.col("horse_finish_time") - pl.col("par_time")).alias("horse_speed_vs_par")
        # )
        #
        # # Calculate the horse_time_vs_winner.
        # feature_df = feature_df.with_columns(
        #     (pl.col("horse_finish_time") - pl.col("win_time")).alias("horse_time_vs_winner")
        # )
        #
        # # Calculate the speed_rating_vs_field.
        # feature_df = feature_df.with_columns(
        #     ((pl.col("speed_rating").sum() - pl.col("speed_rating")) / (pl.len() - 1))
        #     .over(["race_date", "track_code", "race_number"])
        #     .alias("field_avg_speed_rating")
        # )
        #
        # feature_df = feature_df.with_columns(
        #     (pl.col("speed_rating") - pl.col("field_avg_speed_rating")).alias("speed_rating_vs_field")
        # )
        #
        # # Calculate the speed_rating_vs_winner.
        # feature_df = feature_df.with_columns(
        #     pl.col("speed_rating")
        #     .get(pl.col("official_final_position").arg_min())
        #     .over(["race_date", "track_code", "race_number"])
        #     .alias("speed_rating_winner")
        # )
        #
        # feature_df = feature_df.with_columns(
        #     (pl.col("speed_rating") - pl.col("speed_rating_winner")).alias("speed_rating_vs_winner")
        # )

        # Create lag races for horses.
        feature_df = self._process_lag_races(feature_df=feature_df)

        # Process opponent horses.
        feature_df = self._process_opponents(base_df=feature_df)

        # Generate target columns.
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

        # Select only training columns + some helper columns for downstream logic.
        feature_df = feature_df.select(["race_date", "race_number", *self.train_features])

        # Final sort for redundancy.
        feature_df = feature_df.sort(["race_date", "track_code", "race_number"])

        if predict_df is None:
            self.processed_df = feature_df
            return True

        return feature_df

    def _handle_missing_values(self) -> bool:
        base_df: pl.DataFrame = self.processed_df

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

        self.processed_df = base_df

        return True

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
        path_to_historicals_csv = Path.cwd() / "datasets" / "temp_dataset.csv"

        with open(path_to_predict_yaml, "r") as r_yaml:
            predict_data = yaml.safe_load(r_yaml)

        base_df = polars.from_dict({**predict_data["current_race"], **predict_data["current_horse"]})
        base_df = base_df.cast(
            {col: pl.Utf8 for col in base_df.columns if col not in ["race_date", "last_pp_race_date"]}
        )

        load_df = polars.read_csv(path_to_historicals_csv)
        base_df = polars.concat([load_df, base_df])

        base_df = self._build_features(predict_df=base_df)

        return base_df
