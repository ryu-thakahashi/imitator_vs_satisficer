import pymc as pm
import numpy as np
import polars as pl
import pandas as pd
from icecream import ic
import arviz as az
from multiprocessing import freeze_support
from pilot_experiment.data_paths import PROCESSED_DATA_PATH
from pilot_experiment.save_results import save_res, save_fig, save_graphviz

if __name__ == "__main__":
    freeze_support()

    # データ読み込みと前処理
    df = (
        pl.read_parquet(PROCESSED_DATA_PATH / "longer_data.parquet")
        .with_columns(
            [
                pl.when(pl.col("action") == "cooperate")
                .then(1)
                .otherwise(0)
                .alias("action"),
                # bc_ratio は 4→0, その他→1 にするなど、二値化済みと仮定
                pl.when(pl.col("bc_ratio") == 4).then(0).otherwise(1).alias("bc_ratio"),
                # information_condition を0,1,2に変換済みと仮定
                pl.when(pl.col("information_condition") == "summary_information")
                .then(0)
                .when(pl.col("information_condition") == "behavior_only")
                .then(1)
                .when(pl.col("information_condition") == "full_information")
                .then(2)
                .alias("information_condition"),
            ]
        )
        .to_pandas()
    )

    # 階層数の設定（問題文からの定義）
    n_bc = 2
    set_per_bc = 5
    n_set = n_bc * set_per_bc  # = 10
    information_per_set = 3
    n_information = n_set * information_per_set  # 30
    round_per_information = 15
    n_round = n_information * round_per_information  # 450

    # ここから階層IDを作る
    # 1. set_idごとのbc_ratioを取得し、set_to_bcを作る
    # set_id列が存在すると仮定
    # set_idをユニーク化して0~9にマッピング
    unique_sets = np.sort(df["set_id"].unique())
    set_mapping = {sid: i for i, sid in enumerate(unique_sets)}
    # setごとにbc_ratioを一意に取得 (各setに1つのbc_ratioがあると仮定)
    set_bc_df = df.groupby("set_id", as_index=False)["bc_ratio"].first()
    # set_idをset_idxに変換
    set_bc_df["set_idx"] = set_bc_df["set_id"].map(set_mapping)
    # set_idxでソート
    set_bc_df = set_bc_df.sort_values("set_idx")
    # set_to_bcは長さ10の配列
    set_to_bc = set_bc_df["bc_ratio"].to_numpy().astype(int)

    # 2. information_idを作る
    # 各set内にinformation_conditionが3つあると仮定
    # 組み合わせ (set_id, information_condition) をユニーク化して0~29にマッピング
    info_df = df.groupby(
        ["set_id", "information_condition"], as_index=False
    ).size()  # 存在する組み合わせだけ抽出
    # set_id→set_idx 変換
    info_df["set_idx"] = info_df["set_id"].map(set_mapping)
    # 情報IDを割り当て (set_idx, information_condition) の組を並べてID付与
    # ソートしてからID付与
    info_df = info_df.sort_values(["set_idx", "information_condition"])
    info_df["information_idx"] = np.arange(len(info_df))
    # 全部で30個あるはず
    assert len(info_df) == n_information, "information数が想定と異なります。"

    # information_to_set: 情報毎にどのset_idxか
    information_to_set = info_df["set_idx"].to_numpy().astype(int)

    # 3. round_idを作る
    # 各 (set_id, information_condition) に15回のround_numberがあると仮定
    # (set_id, information_condition, round_number)でユニークな組を取得
    round_df = df.groupby(
        ["set_id", "information_condition", "round_number"], as_index=False
    ).size()
    # set_id→set_idx, information_conditionからinformation_idxを決定
    # information_idx検索用に辞書を作る
    info_map = {
        (row["set_id"], row["information_condition"]): row["information_idx"]
        for _, row in info_df.iterrows()
    }

    round_df["set_idx"] = round_df["set_id"].map(set_mapping)
    round_df["information_idx"] = round_df.apply(
        lambda r: info_map[(r["set_id"], r["information_condition"])], axis=1
    )
    # ソートしてround_idx付与
    round_df = round_df.sort_values(["information_idx", "round_number"])
    round_df["round_idx"] = np.arange(len(round_df))
    assert len(round_df) == n_round, "round数が想定と異なります。"

    # round_to_information: 各roundが属するinformationのインデックス配列（長さ450）
    round_to_information = round_df["information_idx"].to_numpy().astype(int)

    # 4. participant_idを0~(n_participants-1)にマッピング
    unique_participants = np.sort(df["participant_id"].unique())
    participant_mapping = {pid: i for i, pid in enumerate(unique_participants)}
    n_participants = len(unique_participants)

    # 各participantがどのround_idxに属するかが必要
    # ここで「各participantは1つのroundのみ」という仮定をおいているが、
    # 実際には複数のroundにまたがる可能性が高い。この場合、モデル構造を変更し、
    # participantとroundの関係を見直す必要がある。
    # ひとまず最初の登場roundでparticipantを割り当てる例（非現実的な例）
    part_round_df = df.groupby("participant_id", as_index=False)[
        ["set_id", "information_condition", "round_number"]
    ].first()
    part_round_df["information_idx"] = part_round_df.apply(
        lambda r: info_map[(r["set_id"], r["information_condition"])], axis=1
    )
    # (information_idx, round_number)からround_idxを求めるため、round_dfを利用
    round_map = {
        (row["information_idx"], row["round_number"]): row["round_idx"]
        for _, row in round_df.iterrows()
    }
    part_round_df["round_idx"] = part_round_df.apply(
        lambda r: round_map[(r["information_idx"], r["round_number"])], axis=1
    )
    # participant_id→participant_idx
    part_round_df["participant_idx"] = part_round_df["participant_id"].map(
        participant_mapping
    )
    # individual_to_round: participantごとのround_idx (長さ=n_participants)
    # participant_idxでソート
    part_round_df = part_round_df.sort_values("participant_idx")
    individual_to_round = part_round_df["round_idx"].to_numpy().astype(int)

    # 5. decisionごとのindividual_idx
    # 各観測行に対して、参加者のparticipant_idxを割り当て
    df["participant_idx"] = df["participant_id"].map(participant_mapping)
    decision_to_individual = df["participant_idx"].to_numpy().astype(int)

    # 観測データ
    decisions = df["action"].to_numpy()

    # ic(set_to_bc, information_to_set, round_to_information, individual_to_round)

    # モデル構築
    with pm.Model() as model:
        mu_global = pm.Normal("mu_global", mu=0.5, sigma=20)

        # BCレベル
        sigma_bc = pm.HalfNormal("sigma_bc", sigma=1)
        mu_bc = pm.Normal("mu_bc", mu=mu_global, sigma=sigma_bc, shape=n_bc)

        # SETレベル
        sigma_set = pm.HalfNormal("sigma_set", sigma=1)
        mu_set = pm.Normal("mu_set", mu=mu_bc[set_to_bc], sigma=sigma_set, shape=n_set)

        # INFORMATIONレベル
        sigma_information = pm.HalfNormal("sigma_information", sigma=1)
        mu_information = pm.Normal(
            "mu_information",
            mu=mu_set[information_to_set],
            sigma=sigma_information,
            shape=n_information,
        )

        # ROUNDレベル
        sigma_round = pm.HalfNormal("sigma_round", sigma=1)
        mu_round = pm.Normal(
            "mu_round",
            mu=mu_information[round_to_information],
            sigma=sigma_round,
            shape=n_round,
        )

        # INDIVIDUALレベル
        sigma_individual = pm.HalfNormal("sigma_individual", sigma=1)
        mu_individual = pm.Normal(
            "mu_individual",
            mu=mu_round[individual_to_round],
            sigma=sigma_individual,
            shape=n_participants,
        )

        # DECISION (観測) レベル
        sigma_obs = pm.HalfNormal("sigma_behavior", sigma=1)
        behavior_obs = pm.Normal(
            "behavior_obs",
            mu=mu_individual[decision_to_individual],
            sigma=sigma_obs,
            observed=decisions,
        )

        trace = pm.sample(draws=2000, tune=1000, target_accept=0.9)

    save_graphviz(model, "hierarichal_bayes")

    az.plot_trace(
        trace,
        var_names=[
            "mu_global",
            "mu_bc",
            "mu_set",
            "mu_information",
            "mu_round",
            "mu_individual",
        ],
    )
    save_fig("hierarchy_bayes_trace")

    summary = pm.summary(
        trace,
        var_names=[
            "mu_global",
            # "sigma_bc",
            "mu_bc",
            # "sigma_set",
            "mu_set",
            # "sigma_information",
            "mu_information",
            # "sigma_round",
            "mu_round",
            # "sigma_individual",
            "mu_individual",
            # "sigma_behavior",
            # "behavior_obs",
        ],
    )
    save_res("hierarchy_bayes.txt", summary)
