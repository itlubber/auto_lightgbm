# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:03:09 2020

@author: meizihang
"""

import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import toad
import lightgbm as lgb

from .utils.logger import logger
from .utils.metrics import solveIV, sloveKS, slovePSI
from .utils.methods import feature_select, auto_choose_params, auto_delete_vars


class auto_lightgbm(object):
    
    def __init__(self, datasets, target="target", var_names=None, params={}, early_stopping_rounds=10, fill_nan_with=-1, min_data=50):
        """
        :param datasets: dict, 数据集字典 {"dev": dev, "oot": oot}
        :param target: 分类标签
        :param var_names: 变量名列表
        :param params: lightgbm输入参数
        :param early_stopping_rounds: lightgbm早停轮次
        :param fill_nan_with: 缺失值填充值
        """
        if "dev" not in datasets: raise NameError("缺少开发样本")
        if "oot" not in datasets: raise NameError("缺少oot样本")
        if target not in datasets["dev"]: raise NameError("标签不在开发样本中")
        if target not in datasets["oot"]: raise NameError("标签不在oot样本中")

        if var_names:
            var_names = [i for i in var_names if i != target]
        else:
            var_names = datasets["dev"].columns.drop(target).tolist()

        datasets["dev"] = datasets["dev"].fillna(fill_nan_with)
        datasets["oot"] = datasets["oot"].fillna(fill_nan_with)

        self.target = target
        self.params = params
        self.datasets = datasets
        self.var_names = var_names
        self.early_stopping_rounds = early_stopping_rounds
        
        if min_data >= 1:
            self.min_data = max(int(len(self.datasets.get("dev", "")) * 0.01), min_data)
        else:
            self.min_data = int(len(self.datasets.get("dev", "")) * max(min_data, 0.01))

    def train(self, select_feature=False, single_delete=True, target=False, select_type="shap", imp_threhold=0, corr_threhold=0.7, psi_threhold=0.1, params_weight=0.2):
        """
        :param no_select: boolean， 是否开启变量筛选
        :param single_delete: boolean， # 是否逐步特征删除
        :param target: string， 自动调参方法["weight","ootks"," avg"], else尝试减少模型过拟合倾向
                "ootks": ootks
                "minus": 1 - abs(devks - ootks)
                "avg": (devks + ootks) / 2
                "weight": ootks + abs(ootks - devks) * 0.2
        :param imp_threhold: float，特征重要性阈值，"shap" 或 "feature_importance"都需要大于等于该阈值
        :param corr_threhold: float, 相关性阈值
        :param psi_threhold: float, 单变量PSI阈值上限
        :param params_weight: float, weight调参方法的权重
        :return: lightgbm.Booster
        """

        logger.info("开始自动建模...")
        logger.info("-" * 50)

        # 获取变量名和数据集
        var_names = self.var_names.copy()

        dev_data = self.datasets.get("dev", "")[var_names + [self.target]]
        oot_data = self.datasets.get("oot", "")[var_names + [self.target]]

        # 未指定参数使用祖传参数
        params = {
            "boosting_type": self.params.get("boosting_type", "gbdt"),
            "objective": self.params.get("objective", "binary"),
            "metric": self.params.get("metric", "auc"),
            "reg_lambda": self.params.get("reg_lambda", 3),
            "reg_alpha": self.params.get("reg_alpha", 0.85),
            "num_leaves": self.params.get("num_leaves", 31),
            "learning_rate": self.params.get("learning_rate", 0.02),
            "min_data": self.params.get("min_data", self.min_data),
            "min_hessian": self.params.get("min_hessian", 0.05),
            "num_threads": self.params.get("num_threads", 1),
            "feature_fraction": self.params.get("feature_fraction", 0.9),
            "bagging_fraction": self.params.get("bagging_fraction", 0.8),
            "bagging_freq": self.params.get("bagging_freq", 2),
            "verbose": self.params.get("verbose", -1),
            "num_boost_round": self.params.get("num_boost_round", 100),
        }

        # base model
        model = lgb.train(params=params, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False, train_set=lgb.Dataset(dev_data[var_names], dev_data[self.target]), valid_sets=lgb.Dataset(oot_data[var_names], oot_data[self.target]))

        # 变量筛选
        if select_feature:
            var_names = feature_select(self.datasets, model, var_names, self.target, select_type, imp_threhold, corr_threhold, psi_threhold)

        # 自动调参
        if target:
            params = auto_choose_params(self.datasets, var_names, self.target, self.min_data, params_weight, early_stopping_rounds=self.early_stopping_rounds, params=params, target=target)
            logger.info("-" * 50)

        # 逐个特征删除
        if single_delete:
            _, var_names = auto_delete_vars(self.datasets, var_names, self.target, self.min_data, self.early_stopping_rounds, params=params)
            logger.info("-" * 50)

        # final model
        model = lgb.train(params=params, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False, train_set=lgb.Dataset(dev_data[var_names], dev_data[self.target]), valid_sets=lgb.Dataset(oot_data[var_names], oot_data[self.target]))

        # 计算KS和PSI
        devks = sloveKS(model, dev_data[var_names], dev_data[self.target])
        ootks = sloveKS(model, oot_data[var_names], oot_data[self.target])
        ootpsi = slovePSI(model, dev_data[var_names], oot_data[var_names])
        dic = {"devks": devks, "ootks": ootks, "ootpsi": ootpsi}
        logger.info(f"KS & PSI: {dic}")
        logger.info("-" * 50)
        logger.info("AutoML建模完成")

        return model, var_names
