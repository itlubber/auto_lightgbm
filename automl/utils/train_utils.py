# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:07:12 2020

@author: meizihang
"""

from tqdm import tqdm
import numpy as np
import shap
import pandas as pd
import lightgbm as lgb
import toad

import hyperopt
from hyperopt import hp
from bayes_opt import BayesianOptimization

from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from .eva_utils import sloveKS, slovePSI


# 特征选择
def feature_select(datasets, model, var_names, dep, select_type='shap', imp_threhold=0, corr_threhold=0.7, psi_threhold=0.1):
    dev_data = datasets.get("dev", "")
    off_data = datasets.get("off", "")

    # shap筛选特征
    if select_type == 'shap':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dev_data[var_names])
        importance = dict(zip(var_names, np.mean([i for i in map(abs,shap_values[1])],axis=0)))
        keep_list = []
        for (key, values) in importance.items():
            if values > imp_threhold:
                keep_list.append(key)
        print("Shap阈值", imp_threhold)
        print("shap删除特征个数：", len(var_names) - len(keep_list),
              "shap保留特征个数：", len(keep_list))
        print('-'*50)
        var_names = keep_list.copy()

    # feature_importance筛选特征
    elif select_type == 'feature_importance':
        importance = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))
        keep_list = []
        for (key, values) in importance.items():
            if values > imp_threhold:
                keep_list.append(key)
        print("feature_importance阈值", imp_threhold)
        print("feature_importance删除特征个数：", len(var_names) - len(keep_list),
              "feature_importance保留特征个数：", len(keep_list))
        print('-'*50)
        var_names = keep_list.copy()

    if corr_threhold:
        dev_slct, drop_lst = toad.selection.select(dev_data[var_names], dev_data[dep], empty=1,
                                                   iv=0, corr=corr_threhold, return_drop=True)
        print("相关性阈值:", corr_threhold,
              "相关性删除特征个数：", len(drop_lst['corr']),
              "相关性保留特征个数：", len(var_names) - len(drop_lst['corr']))
        print('-'*50)
        for i in drop_lst['corr']:
            try:
                var_names.remove(i)
            except:
                continue

    if psi_threhold:
        psi_df = toad.metrics.PSI(dev_data[var_names], off_data[var_names]).sort_values(0)
        psi_df = psi_df.reset_index()
        psi_df = psi_df.rename(columns={'index': 'feature', 0: 'psi'})
        psi = list(psi_df[psi_df.psi < psi_threhold].feature)
        print("PSI阈值", psi_threhold)
        print("PSI删除特征个数：", len(var_names) - len(psi),
              "PSI保留特征个数：", len(psi))
        print('-'*50)
        var_names = [i for i in var_names if i in psi]
    return var_names

# 自动化调参的目标函数
def target_value(target, devks, offks, params_weight):
    if target == "offks":
        return offks
    elif target == "avg":
        return (devks + offks) / 2
    elif target == "weight":
        return offks - abs(devks - offks) * params_weight
    elif target == 'minus':
        return 1 - abs(devks - offks)
    else:
        return devks


# 判断调参后目标函数是否有优化
def check_params(dev_data, off_data, var_names, dep, params,
                 param, train_number, step, target, targetks, params_weight):
    while True:
        try:
            if params[param] + step > 0:
                params[param] += step
                model = lgb.train(params=params, verbose_eval=False,
                                  train_set=lgb.Dataset(dev_data[var_names], dev_data[dep]),
                                  valid_sets=lgb.Dataset(off_data[var_names], off_data[dep]))
                devks = sloveKS(model, dev_data[var_names], dev_data[dep])
                offks = sloveKS(model, off_data[var_names], off_data[dep])
                train_number += 1
                targetks_n = target_value(target=target, devks=devks, offks=offks, params_weight=params_weight)
                if targetks < targetks_n:
                    print("(Good) train_number: %s, devks: %s, offks: %s, params: %s" % (
                        train_number, devks, offks, params))
                    targetks = targetks_n
                else:
                    # print("(Bad) train_number: %s, devks: %s, offks: %s, params: %s" % (
                    # train_number, devks, offks, params))
                    break
            else:
                break
        except:
            break
        params[param] -= step
    return params, targetks, train_number

# 从祖传参数出发，开始找最佳参数
def auto_choose_params(datasets, var_names, dep, min_data, params_weight, early_stopping_rounds=10, params={}, target="weight"):
    """
    :param target:
            "offks": offks最大化;
            "minus": 1-abs(devks-offks) 最大化;
            "avg": (devks+offks)/2  最大化
            "weight": offks + abs(offks - devks) * 0.2 最大化
            "bayes":基于lgb.cv的贝叶斯优化
    :return: 输出最优模型变量
    """
    print("开始参数搜索,目标函数", target)

    dev_data = datasets.get("dev", "")
    off_data = datasets.get("off", "")

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
        "num_boost_round": params.get("num_boost_round", 100)}

    if target == 'bayes':
        hp_dataset = lgb.Dataset(dev_data[var_names], dev_data[dep],silent = True)

        # 交叉验证
        def lgb_cv(feature_fraction, bagging_fraction, min_data_in_leaf, max_depth, min_split_gain, num_leaves,
                   lambda_l1, lambda_l2, num_iterations=100):
            params = {'objective': 'binary', 'num_iterations': num_iterations, 'early_stopping_round': 10,
                      'metric': 'l1','verbose' : -1}
            params['feature_fraction'] = max(min(feature_fraction, 1), 0)
            params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
            params["min_data_in_leaf"] = int(round(min_data_in_leaf))
            params['max_depth'] = int(round(max_depth))
            params['min_split_gain'] = min_split_gain
            params["num_leaves"] = int(round(num_leaves))
            params['lambda_l1'] = max(lambda_l1, 0)
            params['lambda_l2'] = max(lambda_l2, 0)

            cv_result = lgb.cv(params, hp_dataset, nfold=3, seed=328, stratified=True, shuffle=True, verbose_eval=False,
                               early_stopping_rounds=10, metrics='auc')
            return -(min(cv_result['auc-mean']))

        lgb_bo = BayesianOptimization(
            lgb_cv,
            {'feature_fraction': (0.5, 1),
             'bagging_fraction': (0.5, 1),
             'min_data_in_leaf': (1, 100),
             'max_depth': (3, 15),
             'min_split_gain': (0, 5),
             'num_leaves': (16, 128),
             'lambda_l1': (0, 100),
             'lambda_l2': (0, 100),},
            random_state=328,
            verbose=1
        )

        lgb_bo.maximize(init_points=28, n_iter=100)  # init_points表示初始点，n_iter代表迭代次数（即采样数）

        qe = lgb_bo.max['params']

        params['bagging_fraction'] = round(qe['bagging_fraction'], 2)
        params['feature_fraction'] = round(qe['feature_fraction'], 2)
        params['lambda_l1'] = round(qe['lambda_l1'], 2)
        params['lambda_l2'] = round(qe['lambda_l2'], 2)
        params['min_split_gain'] = round(qe['min_split_gain'], 2)

        params['num_leaves'] = int(qe['num_leaves'])
        params['max_depth'] = int(qe['max_depth'])
        params['min_data_in_leaf'] = int(qe['min_data_in_leaf'])

        return params

    model = lgb.train(params=params, verbose_eval=False,early_stopping_rounds=early_stopping_rounds,
                          train_set=lgb.Dataset(dev_data[var_names], dev_data[dep]),
                          valid_sets=lgb.Dataset(off_data[var_names], off_data[dep]))
    
    devks = sloveKS(model, dev_data[var_names], dev_data[dep])
    offks = sloveKS(model, off_data[var_names], off_data[dep])
    train_number = 0
    print("train_number: %s, devks: %s, offks: %s, params: %s" % (train_number, devks, offks, params))
    dic = {"reg_lambda": [10, 2, 1, -1, -2, -10],
           "reg_alpha": [0.5, 0.05, 0.01, -0.01, -0.05, 0.5],
           "num_leaves": [100, 20, 5, -5, -20, -100],
           "learning_rate": [0.2, 0.1, 0.01, 0.001,-0.001, -0.01, -0.1, -0.2],
           "min_data": [min_data * 10, min_data, int(min_data * 0.1), -int(min_data * 0.1),
                        -(min_data), -(min_data * 10)],
           "min_hessian": [0.1, 0.01, -0.01, -0.1],
           "feature_fraction": [0.2, 0.05, -0.05, -0.2],
           "bagging_fraction": [0.15, 0.05, -0.05, -0.15],
           "bagging_freq": [10, 1, -1, -10],
           "num_boost_round": [500, 100, 20, -20, -100, -500]}
    targetks = target_value(target, devks, offks, params_weight)

    while True:
        targetks_lis = []
        for (key, values) in dic.items():
            for v in values:
                if v + params[key] > 0:
                    params, targetks, train_number = check_params(dev_data, off_data, var_names,
                                                                  dep, params, key, train_number,
                                                                  v, target, targetks, params_weight)
                    targetks_n = target_value(target, devks, offks, params_weight)
                    if targetks < targetks_n:
                        targetks_lis.append(targetks)
        print("-" * 50)
        if not targetks_lis:
            break
    print("Best params: ", params)
    return params

# 逐个特征剔除，判断KS是否有降低，尽可能减少模型中的变量个数
def auto_delete_vars(datasets, var_names, dep, min_data, early_stopping_rounds, params={}):
    print("开始逐步删除特征 \t")
    dev_data = datasets.get("dev", "")
    off_data = datasets.get("off", "")
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
        "num_boost_round": params.get("num_boost_round", 100)}

    model = lgb.train(params=delete_params, early_stopping_rounds=early_stopping_rounds, verbose_eval=False,
                      train_set=lgb.Dataset(dev_data[var_names], dev_data[dep]),
                      valid_sets=lgb.Dataset(off_data[var_names], off_data[dep]))
    offks = sloveKS(model, off_data[var_names], off_data[dep])
    train_number, oldks, del_list = 0, offks, list()
    print("train_number: %s, offks: %s" % (train_number, offks))

    while True:
        flag = True
        for var_name in tqdm(var_names):
            names = [var for var in var_names if var_name != var]
            model = lgb.train(params=params, early_stopping_rounds=early_stopping_rounds, verbose_eval=False,
                              train_set=lgb.Dataset(dev_data[names], dev_data[dep]),
                              valid_sets=lgb.Dataset(off_data[names], off_data[dep]))
            train_number += 1
            offks = sloveKS(model, off_data[names], off_data[dep])
            if offks >= oldks:
                oldks = offks
                flag = False
                del_list.append(var_name)
                print("(Good) train_n: %s, offks: %s by vars: %s" % (train_number, offks, var_name))
                var_names = names
            # else:
            # print("(Bad) train_n: %s, offks: %s by vars: %s" % (train_number, offks, len(self.var_names)))
        if flag:
            break
    print("(End) train_n: %s, offks: %s del_list_vars: %s" % (train_number, offks, del_list))
    for i in del_list:
        try:
            var_names.remove(i)
        except:
            continue

    print("逐步删除特征个数：", len(del_list),
          "逐步保留特征个数：", len(var_names))

    return del_list, var_names












































