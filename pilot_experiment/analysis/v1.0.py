import pymc as pm
import numpy as np
import polars as pl
import pandas as pd
from icecream import ic
from multiprocessing import freeze_support
from pilot_experiment.data_paths import PROCESSED_DATA_PATH
from pilot_experiment.save_results import save_res, save_fig, save_graphviz


def get_unique_values(df: pd.DataFrame, col_name: str):
    ic(col_name)
    unique_sets = np.sort(df[col_name].unique())
    set_mapping = {sid: i for i, sid in enumerate(unique_sets)}
    # ic(set_mapping)
    return list(set_mapping.values())

    # from_to_df = (
    #     df.groupby(from_col)[to_col]
    #     .agg("first")
    #     .to_frame()
    #     .reset_index()
    #     .sort_values(from_col)
    # )
    # from_to_df["idx"] = from_to_df[from_col].map(set_mapping)
    # ic(from_to_df)
    # return from_to_df[to_col].to_numpy().astype(int)


if __name__ == "__main__":
    freeze_support()
    # np.random.seed(42)
    df = (
        pl.read_parquet(PROCESSED_DATA_PATH / "longer_data.parquet")
        .with_columns(
            pl.when(pl.col("action") == "cooperate")
            .then(1)
            .otherwise(0)
            .alias("action")
        )
        .to_pandas()
    )

    # b/c，set, information, round, individual, decision の設定
    n_bc = 2
    set_per_bc = 5
    n_set = n_bc * set_per_bc

    information_per_set = 3
    n_information = n_set * information_per_set

    round_per_information = 15
    n_round = n_information * round_per_information

    n_pariticipants = df["participant_id"].nunique()
    n_decision = df.shape[0]

    # 各階層ごと所属配列
    # セットIDのユニーク値取得とマッピング
    unique_sets = np.sort(df["set_id"].unique())
    set_mapping = {sid: i for i, sid in enumerate(unique_sets)}  # 0~9になるはず

    # bc_ratioはセットごとに一意と仮定
    set_bc_df = df.groupby("set_id", as_index=False)[
        "bc_ratio"
    ].first()  # 各セットに対して1つのbc_ratio取得
    # set_idをset_idxに変換
    set_bc_df["set_idx"] = set_bc_df["set_id"].map(set_mapping)
    # set_idxでソート
    set_bc_df = set_bc_df.sort_values("set_idx")
    # set_to_bcは長さ10の配列
    set_to_bc = set_bc_df["bc_ratio"].to_numpy().astype(int)

    # information_to_set = get_unique_values(df, "set_id")
    
    round_to_information = get_unique_values(df, "information_condition")
    individual_to_round = get_unique_values(df, "round_number")
    decision_to_individual = get_unique_values(df, "participant_id")
    ic(
        set_to_bc,
        information_to_set,
        round_to_information,
        individual_to_round,
        # decision_to_individual,
    )

    # 観測データ
    decisions = df["action"].to_numpy()
    ic(decisions)

    # モデル構築
    model = pm.Model()
    with model:
        # b/c 条件レベル
        mu_global = pm.Normal("mu_global", mu=0.5, sigma=20)
        sigma_bc = pm.HalfNormal("sigma_bc", sigma=1)
        mu_bc = pm.Normal(
            "mu_bc",
            mu=mu_global,
            sigma=sigma_bc,
            shape=n_bc,
        )
        ic(mu_bc)

        # set 条件レベル
        sigma_set = pm.HalfNormal("sigma_set", sigma=1)
        mu_set = pm.Normal(
            "mu_set",
            mu=mu_bc[set_to_bc],
            sigma=sigma_set,
            shape=n_set,
        )
        ic(mu_set)

        # information 条件レベル
        sigma_information = pm.HalfNormal("sigma_information", sigma=1)
        mu_information = pm.Normal(
            "mu_information",
            mu=mu_set[information_to_set],
            sigma=sigma_information,
            shape=n_information,
        )
        ic(mu_information)

        # round 条件レベル
        sigma_round = pm.HalfNormal("sigma_round", sigma=1)
        mu_round = pm.Normal(
            "mu_round",
            mu=mu_information[round_to_information],
            sigma=sigma_round,
            shape=n_round,
        )
        ic(mu_round)

        # individual 条件レベル
        sigma_individual = pm.HalfNormal("sigma_individual", sigma=1)
        mu_individual = pm.Normal(
            "mu_individual",
            mu=mu_round[individual_to_round],
            sigma=sigma_individual,
            shape=n_pariticipants,
        )
        ic(mu_individual)

        # decision レベル
        sigma_obs = pm.HalfNormal("sigma_decision", sigma=1)
        behavior_obs = pm.Normal(
            "score_obs",
            mu=mu_individual[decision_to_individual],
            sigma=sigma_obs,
            observed=decisions,
        )
        ic(behavior_obs)

        # MCMCサンプリング
        trace = pm.sample(draws=2000, tune=1000, target_accept=0.9)

    summary = pm.summary(
        trace,
        var_names=[
            "mu_global",
            "sigma_bc",
            "mu_bc",
            "sigma_set",
            "mu_set",
            "sigma_information",
            "mu_information",
            "sigma_round",
            "mu_round",
            "sigma_individual",
            "mu_individual",
            "sigma_decision",
        ],
    )
    save_res(summary, "hierarchy_bayes")

    save_graphviz(model, "simulaiton_model")
