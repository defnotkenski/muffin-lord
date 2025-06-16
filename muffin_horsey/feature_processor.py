import polars as pl


class FeatureProcessor:
    def __init__(self, df: pl.DataFrame):
        self.base_df = df
        self.processed_df = None

    def _process_lag_races(self, feature_df: pl.DataFrame) -> pl.DataFrame:
        # Find horse's last race based on track code and race number.
        feature_df = feature_df.cast({"last_pp_race_number": pl.Int64})

        feature_df = feature_df.with_columns(pl.col("last_pp_track_code").alias("track_code_recent_0"))

        feature_df = feature_df.join(
            feature_df.select(
                [
                    "race_date",
                    "race_number",
                    "track_code",
                    "horse_name",
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
                ]
            ),
            left_on=["last_pp_race_date", "last_pp_track_code", "last_pp_race_number", "horse_name"],
            right_on=["race_date", "track_code", "race_number", "horse_name"],
            how="left",
            suffix="_recent_0",
        )

        return feature_df

    def extract_features(self) -> None:
        # Set the base or working dataframe.
        feature_df = self.base_df

        # Create new features based on existing columns.
        # Add distance_furlongs column.
        feature_df = feature_df.with_columns((pl.col("distance").cast(pl.Float64) / 100).alias("distance_furlongs"))

        # Add field_size column.
        feature_df = feature_df.with_columns(
            pl.count().over(["race_date", "track_code", "race_number"]).alias("field_size")
        )

        # Add rank_in_odds column.
        feature_df = feature_df.with_columns(
            [pl.col("dollar_odds").cast(pl.Float64), pl.col("race_number").cast(pl.Int64)]
        )

        feature_df = feature_df.with_columns(
            pl.int_range(pl.len())
            .over(["race_date", "track_code", "race_number"], order_by="dollar_odds")
            .alias("rank_in_odds")
        ).sort(["race_date", "track_code", "race_number", "dollar_odds"])

        # Add days_since_last_race column.
        feature_df = feature_df.with_columns(pl.col("last_pp_race_date").str.to_datetime())

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
                .is_in(["1", "2", "3"])
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
        # Calculate race_speed_vs_par.
        feature_df = feature_df.cast({"win_time": pl.Float64, "par_time": pl.Float64})

        feature_df = feature_df.with_columns(
            pl.when(pl.col("par_time") != 0.00)
            .then(pl.col("win_time") - pl.col("par_time"))
            .otherwise(pl.lit(None))
            .alias("race_speed_vs_par")
        )

        # Calculate fav_speed_vs_par.
        # length_seconds = 0.2

        # Create lag races for horses.
        feature_df = self._process_lag_races(feature_df=feature_df)

        # Select columns needed for training.
        feature_df = feature_df.select(
            [
                "track_code",
                "horse_name",
                "race_type",
                "race_purse",
                "distance_furlongs",
                "field_size",
                "course_surface",
                "class_rating",
                "track_conditions",
                "runup_distance",
                "rail_distance",
                "sealed",
                "dollar_odds",
                "rank_in_odds",
                "days_since_last_race",
                "trainer_win_pct",
            ]
        )

        return
