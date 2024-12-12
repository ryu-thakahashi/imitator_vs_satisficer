import polars as pl
import polars.selectors as cs
from pathlib import Path
import os
import json
import numpy as np
from icecream import ic


from base import BaseDataProcesser


class PivotLongerData(BaseDataProcesser):
    def __init__(self) -> None:
        super().__init__()

    def read_raw_data(self, file_name: str):
        with open(self.raw_data_path / file_name, "r") as f:
            data = json.load(f)
        bc_ratio = data["bc_ratio"]
        set_id = data["set_id"]
        session_list = data["session_list"]
        return bc_ratio, set_id, session_list

    def get_coop_neighbor_num(self, int_action_list, pos):
        return sum(self.get_neighbor_action_list(int_action_list, pos))

    def caluclate_assortment(self, p_action, coop_neighbor_num):
        if p_action == 0:
            return -(coop_neighbor_num / 4)
        else:
            return coop_neighbor_num / 4

    def convert_action_list_to_int_list(self, action_list):
        return [1 if action == "cooperate" else 0 for action in action_list]

    def write_data(self, df: pl.DataFrame):
        df.write_parquet(self.intermin_data_path / "longer_data.parquet")
        df.write_csv(self.intermin_data_path / "longer_data.csv")

    def get_neighbor_index(self, p_index: int, network_list: list):
        network_list_len = len(network_list)
        neighbor_index_list = []
        for i in range(-2, 3):
            if i == 0:
                continue
            index = (p_index + i) % network_list_len
            neighbor_index_list.append(index)
        return neighbor_index_list

    def get_neighbor_action_list(self, p_index: int, int_action_list: list):
        neighbor_index_list = self.get_neighbor_index(p_index, int_action_list)
        return [int_action_list[i] for i in neighbor_index_list]

    def get_neighbor_payoff_list(self, p_index: int, payoff_list: list):
        neighbor_index_list = self.get_neighbor_index(p_index, payoff_list)
        return [payoff_list[i] for i in neighbor_index_list]

    def get_p_info(self, decision_data) -> list:
        # ic(decision_data)
        p_id = decision_data["reward_id"]
        seat_id = decision_data["seat_id"]
        p_decided_at = decision_data["decision_time_stamp"]
        p_action_str = decision_data["action"]
        p_action_int = 1 if p_action_str == "cooperate" else 0
        p_payoff = decision_data["payoff"]
        return [p_id, seat_id, p_decided_at, p_action_int, p_payoff]

    def get_round_info(self, round_data) -> list:
        network_list = round_data["network_list"]
        action_list = round_data["action_list"]
        int_action_list = self.convert_action_list_to_int_list(action_list)
        payoff_list = round_data["payoff_list"]
        decision_dict = round_data["seat_participant_dict"]
        return network_list, int_action_list, payoff_list, decision_dict

    def get_max_payoff_strategy(self, int_action_list, payoff_list, p_index):
        neighbor_action_list = self.get_neighbor_action_list(p_index, int_action_list)
        neighbor_payoff_list = self.get_neighbor_payoff_list(p_index, payoff_list)

        all_payoff_list = neighbor_payoff_list + [payoff_list[p_index]]
        all_action_list = neighbor_action_list + [int_action_list[p_index]]

        max_payoff = max(all_payoff_list)
        max_payoff_index = all_payoff_list.index(max_payoff)
        return all_action_list[max_payoff_index]

    def get_round_df(self, round_data):
        columns_list = [
            "round_number",
            "round_start_time",
            "p_id",
            "seat_id",
            "p_decided_at",
            "p_action",
            "p_payoff",
            "p_reaction_time",
            "n1_action",
            "n2_action",
            "n3_action",
            "n4_action",
            "n1_payoff",
            "n2_payoff",
            "n3_payoff",
            "n4_payoff",
            "max_payoff_is_coop",
            "positive_payoff",
            "num_of_coop_neighbor",
            "assortment",
        ]

        round_number = round_data["round_number"]
        round_start_time = round_data["exp_start_time"]
        meta_list = [
            round_number,
            round_start_time,
        ]

        network_list, int_action_list, payoff_list, decision_dict = self.get_round_info(
            round_data
        )
        res_mat = []
        for p_id in network_list:
            base_list = meta_list.copy()

            decision_data = decision_dict[p_id]

            # add participant info [p_id, seat_id, p_decided_at, p_action, p_payoff]
            base_list += self.get_p_info(decision_data)

            # add reaction time
            base_list.append(decision_data["decision_time_stamp"] - round_start_time)

            p_index = network_list.index(p_id)
            # add neighbor action and payoff
            neighbor_action_list = self.get_neighbor_action_list(
                p_index, int_action_list
            )
            neighbor_payoff_list = self.get_neighbor_payoff_list(p_index, payoff_list)
            base_list += neighbor_action_list + neighbor_payoff_list

            # add max payoff strategy
            max_payoff_is_coop = self.get_max_payoff_strategy(
                int_action_list, payoff_list, p_index
            )
            base_list.append(max_payoff_is_coop)

            base_list.append(decision_data["payoff"] > 0)

            # add num of coop neighbor and assortment
            num_of_coop_neighbor = sum(neighbor_action_list)
            assortment = self.caluclate_assortment(
                int_action_list[p_index], num_of_coop_neighbor
            )
            base_list += [num_of_coop_neighbor, assortment]

            res_mat.append(base_list)

        return pl.DataFrame(
            res_mat,
            schema=columns_list,
            orient="row",
        )

    def get_session_df(self, session_data):
        round_list = session_data["round_list"]
        res_df = None
        for round_data in round_list:
            round_df = self.get_round_df(round_data)
            if res_df is None:
                res_df = round_df
            else:
                res_df = res_df.vstack(round_df)
        return res_df

    def get_set_df(self, raw_file_name: str):
        bc_ratio, set_id, session_data_list = self.read_raw_data(raw_file_name)

        res_df = None
        session_dict = {
            "summary_information": "0_summary_information",
            "behavior_only": "1_behavior_only",
            "full_information": "2_full_information",
        }
        reorderd_session_dict = {
            "summary_information": "2_summary_information",
            "behavior_only": "1_behavior_only",
            "full_information": "0_full_information",
        }
        for session_data in session_data_list:
            session_df = self.get_session_df(session_data)

            # add set_id, bc_ratio, session_phase
            session_df = session_df.with_columns(
                pl.Series("set_id", [set_id] * session_df.shape[0]),
                pl.Series("bc_ratio", [bc_ratio] * session_df.shape[0]),
                pl.Series(
                    "session_phase",
                    [session_data["session_phase"]] * session_df.shape[0],
                ),
                pl.Series(
                    "orderd_session_name",
                    [session_dict[session_data["session_phase"]]] * session_df.shape[0],
                ),
                pl.Series(
                    "reorderd_session_name",
                    [reorderd_session_dict[session_data["session_phase"]]]
                    * session_df.shape[0],
                ),
            )

            if res_df is None:
                res_df = session_df
            else:
                res_df = res_df.vstack(session_df)
        return res_df

    def process(self):
        res_df = None
        for raw_file in self.raw_data_files:
            if "questionnaire" in raw_file:
                continue
            set_df = self.get_set_df(raw_file)
            if res_df is None:
                res_df = set_df
            else:
                res_df = res_df.vstack(set_df)

        self.write_data(res_df)


if __name__ == "__main__":
    p_info = PivotLongerData()
    ic(p_info.raw_data_files)
    p_info.process()
