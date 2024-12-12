from icecream import ic
import json
import os
import re
import polars as pl

from pilot_experiment.data_paths import RAW_DATA_PATH, PROCESSED_DATA_PATH


def get_target_json_files():
    json_files = os.listdir(RAW_DATA_PATH)
    ic(json_files)
    target_files = [file for file in json_files if re.match(r"^set.*\.json", file)]
    return target_files


def read_json_data(file_name: str):
    json_path = RAW_DATA_PATH / file_name
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def get_session_data(session_list: list):
    for session in session_list:
        yield session


def get_round_list(round_list: list):
    for round_data in round_list:
        yield round_data


def get_decision_list(decision_list: list):
    for decision_data in decision_list:
        yield decision_data


if __name__ == "__main__":
    empty_list = [None] * 9
    res_matrix = [empty_list for _ in range(5000)]
    ic(len(res_matrix))
    insert_row = 0

    target_json_files = get_target_json_files()
    # json_file_name = target_json_files[0]
    for json_file_name in target_json_files:
        raw_data = read_json_data(json_file_name)

        bc_ratio = raw_data["bc_ratio"]
        set_id = raw_data["set_id"]
        # ic(raw_data)
        ic(bc_ratio, set_id)

        for session_data in get_session_data(raw_data["session_list"]):
            # ic(session_data.keys())
            information_condition = session_data["session_phase"]
            reward_dict = session_data["seat_reward_dict"]
            # ic(session_phase, reward_dict)

            for round_data in get_round_list(session_data["round_list"]):
                # ic(round_data)
                # ic(round_data.keys())
                round_number = round_data["round_number"]
                network_list = round_data["network_list"]
                action_list = round_data["action_list"]
                payoff_list = round_data["payoff_list"]

                population_size = len(network_list)
                for node_i in range(population_size):
                    seat_id = network_list[node_i]
                    participant_id = reward_dict[seat_id]
                    action = action_list[node_i]
                    payoff = payoff_list[node_i]

                    res_matrix[insert_row] = [
                        unique_set_id,
                        bc_ratio,
                        information_condition,
                        round_number,
                        population_size,
                        participant_id,
                        seat_id,
                        action,
                        payoff,
                    ]
                    insert_row += 1

    ic(insert_row)
    ic(res_matrix[:10])
    res_df = pl.DataFrame(
        res_matrix,
        schema=[
            ("set_id", pl.Int16),
            ("bc_ratio", pl.Int8),
            ("information_condition", pl.String),
            ("round_number", pl.Int8),
            ("population_size", pl.Int8),
            ("participant_id", pl.String),
            ("seat_id", pl.Int64),
            ("action", pl.String),
            ("payoff", pl.Int16),
        ],
        orient="row",
    ).drop_nulls()
    ic(res_df)

    res_df.write_csv(PROCESSED_DATA_PATH / "longer_data.csv")
    res_df.write_parquet(PROCESSED_DATA_PATH / "longer_data.parquet")
