import pymc as pm
import numpy as np
from icecream import ic
from multiprocessing import freeze_support
from pilot_experiment.data_paths import PROCESSED_DATA_PATH
from pilot_experiment.save_results import save_res, save_fig, save_graphviz

if __name__ == "__main__":
    freeze_support()
    np.random.seed(42)

    # 階層数の設定（あくまで例）
    n_chiho = 2  # 地方数
    todofuken_per_chiho = 3
    n_todofuken = n_chiho * todofuken_per_chiho

    chiiki_per_todofuken = 4
    n_chiiki = n_todofuken * chiiki_per_todofuken

    school_per_chiiki = 5
    n_schools = n_chiiki * school_per_chiiki

    class_per_school = 2
    n_classes = n_schools * class_per_school

    # students_per_class = 10
    # n_students = n_classes * students_per_class
    n_students = 131

    # 各階層ごと所属配列
    todofuken_to_chiho = np.repeat(np.arange(n_chiho), todofuken_per_chiho)
    chiiki_to_todofuken = np.repeat(np.arange(n_todofuken), chiiki_per_todofuken)
    school_to_chiiki = np.repeat(np.arange(n_chiiki), school_per_chiiki)
    class_to_school = np.repeat(np.arange(n_schools), class_per_school)

    # 真のパラメータ設定（シミュレーション用）
    true_mu_global = 60.0
    true_sigma_chiho = 10.0  # 地方間のばらつき
    true_sigma_todofuken = 7.0
    true_sigma_chiiki = 5.0
    true_sigma_school = 4.0
    true_sigma_class = 3.0
    true_sigma_obs = 8.0  # 生徒間の観測ノイズ

    # 地方ごとの真の平均
    ic(n_chiho)
    true_mu_chiho = np.random.normal(true_mu_global, true_sigma_chiho, size=n_chiho)

    # 都道府県ごとの真の平均
    true_mu_todofuken = []
    for c in range(n_chiho):
        idxs = np.where(todofuken_to_chiho == c)[0]
        for t in idxs:
            true_mu_todofuken.append(
                np.random.normal(true_mu_chiho[c], true_sigma_todofuken)
            )
    true_mu_todofuken = np.array(true_mu_todofuken)

    # 地域ごとの真の平均
    true_mu_chiiki = []
    for t in range(n_todofuken):
        idxs = np.where(chiiki_to_todofuken == t)[0]
        for ck in idxs:
            true_mu_chiiki.append(
                np.random.normal(true_mu_todofuken[t], true_sigma_chiiki)
            )
    true_mu_chiiki = np.array(true_mu_chiiki)

    # 学校ごとの真の平均
    true_mu_school = []
    for ck in range(n_chiiki):
        idxs = np.where(school_to_chiiki == ck)[0]
        for s in idxs:
            true_mu_school.append(
                np.random.normal(true_mu_chiiki[ck], true_sigma_school)
            )
    true_mu_school = np.array(true_mu_school)

    # クラスごとの真の平均
    true_mu_class = []
    for s in range(n_schools):
        idxs = np.where(class_to_school == s)[0]
        for cl in idxs:
            true_mu_class.append(np.random.normal(true_mu_school[s], true_sigma_class))
    true_mu_class = np.array(true_mu_class)

    # インデックスの生成
    class_idx = np.repeat(np.arange(n_classes), n_students)
    # 生徒スコア観測
    scores = np.random.normal(true_mu_class[class_idx], true_sigma_obs)

    with pm.Model() as model:
        # 地方レベル
        mu_global = pm.Normal("mu_global", mu=50, sigma=20)
        sigma_chiho = pm.HalfNormal("sigma_chiho", sigma=10)
        mu_chiho = pm.Normal(
            "mu_chiho",
            mu=mu_global,
            sigma=sigma_chiho,
            shape=n_chiho,
        )

        # 都道府県レベル
        sigma_todofuken = pm.HalfNormal("sigma_todofuken", sigma=10)
        mu_todofuken = pm.Normal(
            "mu_todofuken",
            mu=mu_chiho[todofuken_to_chiho],
            sigma=sigma_todofuken,
            shape=n_todofuken,
        )

        # 地域レベル
        sigma_chiiki = pm.HalfNormal("sigma_chiiki", sigma=10)
        mu_chiiki = pm.Normal(
            "mu_chiiki",
            mu=mu_todofuken[chiiki_to_todofuken],
            sigma=sigma_chiiki,
            shape=n_chiiki,
        )

        # 学校レベル
        sigma_school = pm.HalfNormal("sigma_school", sigma=10)
        mu_school = pm.Normal(
            "mu_school",
            mu=mu_chiiki[school_to_chiiki],
            sigma=sigma_school,
            shape=n_schools,
        )

        # クラスレベル
        sigma_class = pm.HalfNormal("sigma_class", sigma=10)
        mu_class = pm.Normal(
            "mu_class",
            mu=mu_school[class_to_school],
            sigma=sigma_class,
            shape=n_classes,
        )

        # 観測ノイズ
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=10)

        # 観測モデル
        score_obs = pm.Normal(
            "score_obs", mu=mu_class[class_idx], sigma=sigma_obs, observed=scores
        )

        # サンプリング
        trace = pm.sample(draws=2000, tune=1000, target_accept=0.9)

    summary = pm.summary(
        trace,
        var_names=[
            "mu_global",
            "sigma_chiho",
            "mu_chiho",
            "sigma_todofuken",
            "mu_todofuken",
            "sigma_chiiki",
            "mu_chiiki",
            "sigma_school",
            "mu_school",
            "sigma_class",
            "mu_class",
            "sigma_obs",
        ],
    )
    save_res(summary, "simulation_hierarchy_bayes")

    save_graphviz(model, "simulation_hierarchy_bayes")
