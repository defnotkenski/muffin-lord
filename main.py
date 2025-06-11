import xml.etree.ElementTree as ET
import polars
import json
from datetime import datetime


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f"Hi, {name}")  # Press ⌘F8 to toggle the breakpoint.

    with open("tags_selector.json", "r") as r:
        tags_selector = json.load(r)

    horse_count: int = 0
    polars_dict: list[dict] = []

    tree = ET.parse("datasets/2025_06/GP20250601TCH.xml")
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
                parsed_data[key] = race.find(value).text

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

            del parsed_data["jockey_first_name"]
            del parsed_data["jockey_last_name"]

            horse_count += 1
            polars_dict.append({"race_date": race_date, **parsed_data})

    print(f"Num of horses: {horse_count}")
    _xml_polars = polars.from_dicts(polars_dict)

    return


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print_hi("Kenny")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
