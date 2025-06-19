import xml.etree.ElementTree as ET
import polars
import json
from datetime import datetime
from muffin_horsey.feature_processor import FeatureProcessor
from pathlib import Path
from schema import COLUMN_TYPES


def process_xml(xml_path: Path) -> list[dict]:
    with open("tags_selector.json", "r") as r:
        tags_selector = json.load(r)

    polars_dict: list[dict] = []

    # tree = ET.parse("datasets/2025_06/GP20250601TCH.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for race in root.findall("RACE"):

        for entry in race.findall("ENTRY"):
            parsed_data = {}

            # ===== Get date of event. =====

            race_date = root.attrib["RACE_DATE"]
            race_date = datetime.fromisoformat(race_date)

            # ===== Get TRACK data. =====

            track_tags = tags_selector["track"]

            for key, value in track_tags.items():
                parsed_data[key] = root.find(f"TRACK/{value}").text

            # ===== Get RACE data. =====

            race_number = race.attrib["NUMBER"]
            race_tags = tags_selector["race"]

            parsed_data["race_number"] = race_number

            for key, value in race_tags.items():
                race_tag_value = race.find(value)
                parsed_data[key] = race_tag_value.text if race_tag_value is not None else None

            # ===== Get ENTRY data. =====

            entry_tags = tags_selector["entry"]

            for key, value in entry_tags.items():
                tag_value = entry.find(value)
                parsed_data[key] = tag_value.text if tag_value is not None else None

            # Do some manipulation with the Jockey names.
            jockey_first = entry.find("JOCKEY/FIRST_NAME").text
            jockey_last = entry.find("JOCKEY/LAST_NAME").text
            jockey_full_name = jockey_first + " " + jockey_last

            parsed_data["jockey_full_name"] = jockey_full_name

            polars_dict.append({"race_date": race_date, **parsed_data})

    return polars_dict


def _cleanup_dataframe(base_polars_df: polars.DataFrame) -> polars.DataFrame:
    # Convert all instances of " " to None across all columns.
    cols_to_replace = ["pace_call_1", "pace_call_2", "trainer_first_name", "owner_full_name"]

    base_polars_df = base_polars_df.with_columns(
        [
            polars.when(polars.col(col) == " ").then(polars.lit(None)).otherwise(polars.col(col)).alias(col)
            for col in cols_to_replace
        ]
    )

    # Check for single space " " in all columns.
    _df_check_1_post = base_polars_df.select([(polars.col("*").exclude("race_date", "last_pp_race_date") == " ").sum()])

    return base_polars_df


def merge_xml() -> polars.DataFrame:
    all_data = []
    xml_files = Path.cwd().joinpath("datasets").rglob("*.xml")

    # Iterate through every XML file and parse.
    for xml in xml_files:
        file_data = process_xml(xml_path=xml)
        all_data.extend(file_data)

    # Create polars dataframe from dict and apply appropriate sorting.
    polars_df = polars.from_dicts(all_data)

    # Cleanup outlier values.
    polars_df = _cleanup_dataframe(base_polars_df=polars_df)

    # Cast columns to appropriate dtypes.
    polars_df = polars_df.cast(COLUMN_TYPES)
    polars_df = polars_df.with_columns(polars.col("last_pp_race_date").str.to_datetime())

    # Apply appropriate sorting before sending it off.
    polars_df = polars_df.sort(["race_date", "track_code", "race_number", "dollar_odds"])

    return polars_df


if __name__ == "__main__":
    merged_df = merge_xml()

    feature_processor = FeatureProcessor(df=merged_df, target_type="place")
    feature_processor.get_dataframe()
