import yaml
from pathlib import Path


def generate_train_features(lag_count: int, other_count: int) -> list:
    yaml_path = Path.cwd() / "muffin_horsey" / "sample_training.yaml"
    # yaml_path = Path.cwd() / "sample_training.yaml"

    with open(yaml_path) as sample_yaml:
        sample_features = yaml.safe_load(sample_yaml)

    current_race = list(sample_features["current_race"].keys())
    current_horse = list(sample_features["current_horse"].keys())

    master_current_horse_lags = []

    for i in range(lag_count):
        current_horse_lags: list[str] = list(sample_features["current_horse_lags"].keys())
        modified_col_name = [col.replace("recent_0_", f"recent_{i}_") for col in current_horse_lags]

        master_current_horse_lags.extend(modified_col_name)

    master_other_horse = []
    master_other_horse_lags = []

    for other_num in range(other_count):
        other_horse_current_race_metadata: list[str] = list(sample_features["other_horse"].keys())
        modified_race_metadata = [
            horse.replace("other_X_", f"opp_{other_num+1}_") for horse in other_horse_current_race_metadata
        ]

        master_other_horse.extend(modified_race_metadata)

        for lag_num in range(lag_count):
            other_horse_lags: list[str] = list(sample_features["other_horse_lags"].keys())
            modified_other_horse_lags_col = [
                horse.replace("other_X_recent_X_", f"opp_{other_num+1}_recent_{lag_num}_") for horse in other_horse_lags
            ]

            master_other_horse_lags.extend(modified_other_horse_lags_col)

    master_features = [
        *current_race,
        *current_horse,
        *master_current_horse_lags,
        *master_other_horse,
        *master_other_horse_lags,
    ]

    return master_features


if __name__ == "__main__":
    generate_train_features(lag_count=3, other_count=4)
