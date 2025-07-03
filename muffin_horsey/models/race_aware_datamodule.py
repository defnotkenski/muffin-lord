from pytorch_tabular.tabular_datamodule import TabularDatamodule
from torch.utils.data import DataLoader
from functools import partial
from muffin_horsey.models.race_aware_dataset import RaceAwareDataset, race_collate_fn
from muffin_horsey.feature_processor import FeatureProcessor


class RaceAwareDatamodule(TabularDatamodule):
    def __init__(self, *args, max_horses=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_horses = max_horses

    def _cache_dataset(self):
        """Override to create race-aware datasets instead of standard TabularDataset."""

        # Create FeatureProcessor instance to get race-grouped data
        feature_processor = FeatureProcessor(None, self.config.task)  # We'll set the df later

        # Process train data
        feature_processor.processed_df = self.train  # Set the dataframe
        train_races = feature_processor.prepare_race_grouped_data()

        # Process validation data
        feature_processor.processed_df = self.validation if hasattr(self, "validation") else None
        val_races = feature_processor.prepare_race_grouped_data() if hasattr(self, "validation") else []

        # Process test data if exists
        test_races = []
        if hasattr(self, "test") and self.test is not None:
            feature_processor.processed_df = self.test
            test_races = feature_processor.prepare_race_grouped_data()

        # Create race-aware datasets
        self.train_dataset = RaceAwareDataset(
            races_data=train_races,
            continuous_cols=self.config.continuous_cols,
            categorical_cols=self.config.categorical_cols,
            target_col=self.config.target[0],  # Assuming single target
            max_horses=self.max_horses,
        )

        if val_races:
            self.validation_dataset = RaceAwareDataset(
                races_data=val_races,
                continuous_cols=self.config.continuous_cols,
                categorical_cols=self.config.categorical_cols,
                target_col=self.config.target[0],
                max_horses=self.max_horses,
            )

        if test_races:
            self.test_dataset = RaceAwareDataset(
                races_data=test_races,
                continuous_cols=self.config.continuous_cols,
                categorical_cols=self.config.categorical_cols,
                target_col=self.config.target[0],
                max_horses=self.max_horses,
            )

    def train_dataloader(self, batch_size=None):
        """Override to use custom collate function."""
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=partial(race_collate_fn, max_horses=self.max_horses),
            pin_memory=True,
        )

    def val_dataloader(self, batch_size=None):
        """Override to use custom collate function."""
        if not hasattr(self, "validation_dataset"):
            return None

        return DataLoader(
            self.validation_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(race_collate_fn, max_horses=self.max_horses),
            pin_memory=True,
        )

    def test_dataloader(self, batch_size=None):
        """Override to use custom collate function."""
        if not hasattr(self, "test_dataset"):
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=partial(race_collate_fn, max_horses=self.max_horses),
            pin_memory=True,
        )
