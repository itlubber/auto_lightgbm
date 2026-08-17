# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:07:12 2020

@author: meizihang
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    import shap
except ImportError:
    shap = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from hyperopt import hp
except ImportError:
    hp = None

try:
    from bayes_opt import BayesianOptimization
except ImportError:
    BayesianOptimization = None
from hscredit.core.metrics import batch_psi
from hscredit.core.selectors import ScorecardFeatureSelection

from .logger import logger
from .metrics import sloveKS, slovePSI


def _normalize_lgb_params(params):
    normalized = dict(params)
    for key in ("feature_fraction", "bagging_fraction"):
        if key in normalized:
            try:
                value = float(normalized[key])
            except (TypeError, ValueError):
                continue
            normalized[key] = max(min(value, 1.0), 1e-6)
    return normalized


def _train_lgbm(params, train_set, valid_set=None, early_stopping_rounds=None):
    train_params = _normalize_lgb_params(params)
    num_boost_round = int(train_params.pop("num_boost_round", 100))
    callbacks = []

    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    callbacks.append(lgb.log_evaluation(period=0))

    valid_sets = [valid_set] if valid_set is not None else None

    return lgb.train(
        params=train_params,
        train_set=train_set,
        valid_sets=valid_sets,
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )


def feature_select(datasets, model, var_names, dep, select_type="shap", imp_threhold=0, corr_threhold=0.7, psi_threhold=0.1):
    """
    特征选择
    """
    dev_data = datasets.get("dev", "")
    oot_data = datasets.get("oot", "")

    # shap筛选特征
    if select_type == "shap":
        if shap is None:
            logger.info("Shap未安装，跳过shap筛选")
            logger.info("-" * 50)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(dev_data[var_names])
            if isinstance(shap_values, list):
                shap_array = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
            else:
                shap_array = np.asarray(shap_values)

            if shap_array.ndim == 1:
                shap_importance = np.abs(shap_array)
            else:
                shap_importance = np.mean(np.abs(shap_array), axis=0)

            importance = dict(zip(var_names, shap_importance))
            keep_list = []
            for key, values in importance.items():
                if values > imp_threhold:
                    keep_list.append(key)
            if keep_list:
                logger.info(f"Shap阈值 {imp_threhold}")
                logger.info(f"shap删除特征个数：{len(var_names) - len(keep_list)}, shap保留特征个数：{len(keep_list)}")
                logger.info("-" * 50)
                var_names = keep_list.copy()
            else:
                logger.info(f"Shap阈值 {imp_threhold}, 但未保留任何特征，跳过该阶段")
                logger.info("-" * 50)

    # feature_importance筛选特征
    elif select_type == "feature_importance":
        importance = dict(zip(model.feature_name(), model.feature_importance(importance_type="gain")))
        keep_list = []
        for key, values in importance.items():
            if values > imp_threhold:
                keep_list.append(key)
        if keep_list:
            logger.info(f"feature_importance阈值 {imp_threhold}")
            logger.info(f"feature_importance删除特征个数：{len(var_names) - len(keep_list)}, feature_importance保留特征个数：{len(keep_list)}")
            logger.info("-" * 50)
            var_names = keep_list.copy()
        else:
            logger.info(f"feature_importance阈值 {imp_threhold}, 但未保留任何特征，跳过该阶段")
            logger.info("-" * 50)

    # 相关性筛选特征
    if corr_threhold:
        selector = ScorecardFeatureSelection(
            null_threshold=None,
            iv_threshold=None,
            corr_threshold=corr_threhold,
            mode_threshold=None,
            target=dep,
        )
        selector.fit(dev_data[var_names], dev_data[dep])
        keep_set = set(selector.selected_features_)
        drop_lst = [name for name in var_names if name not in keep_set]
        if keep_set:
            logger.info(f"相关性阈值: {corr_threhold}, 相关性删除特征个数: {len(drop_lst)}, 相关性保留特征个数: {len(keep_set)}")
            logger.info("-" * 50)
            var_names = [name for name in var_names if name in keep_set]
        else:
            logger.info(f"相关性阈值: {corr_threhold}, 但未保留任何特征，跳过该阶段")
            logger.info("-" * 50)

    # PSI筛选特征
    if psi_threhold:
        psi_df = batch_psi(dev_data[var_names], oot_data[var_names], features=var_names)
        psi = psi_df.loc[psi_df["PSI"] < psi_threhold, "特征"].tolist()
        if psi:
            logger.info(f"PSI阈值 {psi_threhold}")
            logger.info(f"PSI删除特征个数: {len(var_names) - len(psi)}, PSI保留特征个数: {len(psi)}")
            logger.info("-" * 50)
            var_names = [i for i in var_names if i in psi]
        else:
            logger.info(f"PSI阈值 {psi_threhold}, 但未保留任何特征，跳过该阶段")
            logger.info("-" * 50)
    return var_names


def target_value(target, devks, ootks, params_weight):
    """
    自动化调参的目标函数
    """
    if target == "ootks":
        return ootks
    elif target == "avg":
        return (devks + ootks) / 2
    elif target == "weight":
        return ootks - abs(devks - ootks) * params_weight
    elif target == "minus":
        return 1 - abs(devks - ootks)
    else:
        return devks


def check_params(dev_data, oot_data, var_names, dep, params, param, train_number, step, target, targetks, params_weight):
    """
    判断调参后目标函数是否有优化
    """
    while True:
        try:
            candidate = params[param] + step
            if param in ("feature_fraction", "bagging_fraction"):
                candidate = max(min(candidate, 1.0), 1e-6)
            if candidate > 0 and candidate != params[param]:
                trial_params = dict(params)
                trial_params[param] = candidate
                model = _train_lgbm(trial_params, lgb.Dataset(dev_data[var_names], dev_data[dep]), lgb.Dataset(oot_data[var_names], oot_data[dep]))
                devks = sloveKS(model, dev_data[var_names], dev_data[dep])
                ootks = sloveKS(model, oot_data[var_names], oot_data[dep])
                train_number += 1
                targetks_n = target_value(target=target, devks=devks, ootks=ootks, params_weight=params_weight)
                if targetks < targetks_n:
                    logger.info("(Good) train_number: %s, devks: %s, ootks: %s, params: %s" % (train_number, devks, ootks, trial_params))
                    targetks = targetks_n
                    params = trial_params
                else:
                    # logger.info("(Bad) train_number: %s, devks: %s, ootks: %s, params: %s" % (train_number, devks, ootks, params))
                    break
            else:
                break
        except:
            break
    return params, targetks, train_number


def auto_choose_params(datasets, var_names, dep, min_data, params_weight, early_stopping_rounds=10, params={}, target="weight"):
    """从祖传参数出发，开始找最佳参数
    :param target:
            "ootks": ootks最大化;
            "minus": 1-abs(devks-ootks) 最大化;
            "avg": (devks+ootks)/2  最大化
            "weight": ootks + abs(ootks - devks) * 0.2 最大化
            "bayes":基于lgb.cv的贝叶斯优化
    :return: 输出最优模型变量
    """
    logger.info(f"开始参数搜索,目标函数 {target}")

    dev_data = datasets.get("dev", "")
    oot_data = datasets.get("oot", "")

    params = {
        "boosting_type": params.get("boosting_type", "gbdt"),
        "objective": params.get("objective", "binary"),
        "metric": params.get("metric", "auc"),
        "reg_lambda": params.get("reg_lambda", 3),
        "reg_alpha": params.get("reg_alpha", 0.85),
        "num_leaves": params.get("num_leaves", 31),
        "learning_rate": params.get("learning_rate", 0.02),
        "min_data": params.get("min_data", min_data),
        "min_hessian": params.get("min_hessian", 0.05),
        "num_threads": params.get("num_threads", 1),
        "feature_fraction": params.get("feature_fraction", 0.9),
        "bagging_fraction": params.get("bagging_fraction", 0.8),
        "bagging_freq": params.get("bagging_freq", 2),
        "verbose": params.get("verbose", -1),
        "num_boost_round": params.get("num_boost_round", 100),
    }

    if target == "bayes":
        if hp is None or BayesianOptimization is None:
            raise ImportError("bayes_opt 或 hyperopt 未安装，无法使用 bayes 参数搜索")

        hp_dataset = lgb.Dataset(dev_data[var_names], dev_data[dep], silent=True)

        # 交叉验证
        def lgb_cv(feature_fraction, bagging_fraction, min_data_in_leaf, max_depth, min_split_gain, num_leaves, lambda_l1, lambda_l2, num_iterations=100):
            params = {"objective": "binary", "num_iterations": num_iterations, "early_stopping_round": 10, "metric": "l1", "verbose": -1}
            params["feature_fraction"] = max(min(feature_fraction, 1.), 0)
            params["bagging_fraction"] = max(min(bagging_fraction, 1.), 0)
            params["min_data_in_leaf"] = int(round(min_data_in_leaf))
            params["max_depth"] = int(round(max_depth))
            params["min_split_gain"] = min_split_gain
            params["num_leaves"] = int(round(num_leaves))
            params["lambda_l1"] = max(lambda_l1, 0)
            params["lambda_l2"] = max(lambda_l2, 0)

            cv_result = lgb.cv(
                params,
                hp_dataset,
                num_boost_round=num_iterations,
                nfold=3,
                seed=328,
                stratified=True,
                shuffle=True,
                metrics="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            return -(min(cv_result["auc-mean"]))

        lgb_bo = BayesianOptimization(
            lgb_cv,
            {
                "feature_fraction": (0.5, 1.),
                "bagging_fraction": (0.5, 1.),
                "min_data_in_leaf": (1, 100),
                "max_depth": (3, 15),
                "min_split_gain": (0, 5),
                "num_leaves": (16, 128),
                "lambda_l1": (0, 100),
                "lambda_l2": (0, 100),
            },
            random_state=328,
            verbose=1,
        )

        lgb_bo.maximize(init_points=28, n_iter=100)  # init_points表示初始点，n_iter代表迭代次数（即采样数）

        qe = lgb_bo.max["params"]

        params["bagging_fraction"] = round(qe["bagging_fraction"], 2)
        params["feature_fraction"] = round(qe["feature_fraction"], 2)
        params["lambda_l1"] = round(qe["lambda_l1"], 2)
        params["lambda_l2"] = round(qe["lambda_l2"], 2)
        params["min_split_gain"] = round(qe["min_split_gain"], 2)

        params["num_leaves"] = int(qe["num_leaves"])
        params["max_depth"] = int(qe["max_depth"])
        params["min_data_in_leaf"] = int(qe["min_data_in_leaf"])

        return params

    model = _train_lgbm(params, lgb.Dataset(dev_data[var_names], dev_data[dep]), lgb.Dataset(oot_data[var_names], oot_data[dep]), early_stopping_rounds=early_stopping_rounds)

    devks = sloveKS(model, dev_data[var_names], dev_data[dep])
    ootks = sloveKS(model, oot_data[var_names], oot_data[dep])
    train_number = 0
    logger.info("train_number: %s, devks: %s, ootks: %s, params: %s" % (train_number, devks, ootks, params))
    dic = {
        "reg_lambda": [10, 2, 1, -1, -2, -10],
        "reg_alpha": [0.5, 0.05, 0.01, -0.01, -0.05, 0.5],
        "num_leaves": [100, 20, 5, -5, -20, -100],
        "learning_rate": [0.2, 0.1, 0.01, 0.001, -0.001, -0.01, -0.1, -0.2],
        "min_data": [min_data * 10, min_data, int(min_data * 0.1), -int(min_data * 0.1), -(min_data), -(min_data * 10)],
        "min_hessian": [0.1, 0.01, -0.01, -0.1],
        "feature_fraction": [0.2, 0.05, -0.05, -0.2],
        "bagging_fraction": [0.15, 0.05, -0.05, -0.15],
        "bagging_freq": [10, 1, -1, -10],
        "num_boost_round": [500, 100, 20, -20, -100, -500],
    }
    targetks = target_value(target, devks, ootks, params_weight)

    while True:
        targetks_lis = []
        for key, values in dic.items():
            for v in values:
                if v + params[key] > 0:
                    params, targetks, train_number = check_params(dev_data, oot_data, var_names, dep, params, key, train_number, v, target, targetks, params_weight)
                    targetks_n = target_value(target, devks, ootks, params_weight)
                    if targetks < targetks_n:
                        targetks_lis.append(targetks)
        logger.info("-" * 50)
        if not targetks_lis:
            break
    logger.info("Best params: ", params)
    return params


def auto_delete_vars(datasets, var_names, dep, min_data, early_stopping_rounds, params={}):
    """
    逐个特征剔除，判断KS是否有降低，尽可能减少模型中的变量个数
    """
    logger.info("开始逐步删除特征 \t")
    dev_data = datasets.get("dev", "")
    oot_data = datasets.get("oot", "")
    delete_params = {
        "boosting_type": params.get("boosting_type", "gbdt"),
        "objective": params.get("objective", "binary"),
        "metric": params.get("metric", "auc"),
        "reg_lambda": params.get("reg_lambda", 3),
        "reg_alpha": params.get("reg_alpha", 0.85),
        "num_leaves": params.get("num_leaves", 31),
        "learning_rate": params.get("learning_rate", 0.02),
        "min_data": params.get("min_data", min_data),
        "min_hessian": params.get("min_hessian", 0.05),
        "num_threads": params.get("num_threads", 1),
        "feature_fraction": params.get("feature_fraction", 0.9),
        "bagging_fraction": params.get("bagging_fraction", 0.8),
        "bagging_freq": params.get("bagging_freq", 2),
        "verbose": params.get("verbose", -1),
        "num_boost_round": params.get("num_boost_round", 100),
    }

    model = _train_lgbm(delete_params, lgb.Dataset(dev_data[var_names], dev_data[dep]), lgb.Dataset(oot_data[var_names], oot_data[dep]), early_stopping_rounds=early_stopping_rounds)
    ootks = sloveKS(model, oot_data[var_names], oot_data[dep])
    train_number, oldks, del_list = 0, ootks, list()
    logger.info("train_number: %s, ootks: %s" % (train_number, ootks))

    while True:
        flag = True
        for var_name in tqdm(var_names):
            names = [var for var in var_names if var_name != var]
            if not names:
                continue
            model = _train_lgbm(params, lgb.Dataset(dev_data[names], dev_data[dep]), lgb.Dataset(oot_data[names], oot_data[dep]), early_stopping_rounds=early_stopping_rounds)
            train_number += 1
            ootks = sloveKS(model, oot_data[names], oot_data[dep])
            if ootks >= oldks:
                oldks = ootks
                flag = False
                del_list.append(var_name)
                logger.info("(Good) train_n: %s, ootks: %s by vars: %s" % (train_number, ootks, var_name))
                var_names = names
            # else:
            # logger.info("(Bad) train_n: %s, ootks: %s by vars: %s" % (train_number, ootks, len(self.var_names)))
        if flag:
            break
    logger.info("(End) train_n: %s, ootks: %s del_list_vars: %s" % (train_number, ootks, del_list))
    for i in del_list:
        try:
            var_names.remove(i)
        except:
            continue

    logger.info(f"逐步删除特征个数: {len(del_list)}, 逐步保留特征个数: {len(var_names)}")

    return del_list, var_names
