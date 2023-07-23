# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:03:09 2020

@author: meizihang
"""

import pandas as pd
import numpy as np
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
import os
from sklearn import tree

import warnings
import pandas as pd
import toad
from .utils.Decision_utils import DTR_TO_SQL
from .utils.eva_utils import solveIV
from .utils.train_utils import feature_select, auto_choose_params, auto_delete_vars
from .utils.eva_utils import sloveKS, slovePSI
import lightgbm as lgb

warnings.filterwarnings('ignore')

#lightgbm自动建模
class auto_lightgbm(object):
    def __init__(self, datasets, dep, var_names, uid='Zihang.Mei', params={}, early_stopping_rounds=10 ,fill_nan_with=-1):

        try:
            qe = datasets['dev']
        except:
            raise NameError('缺少开发样本')
        try:
            qe = datasets['dev']
        except:
            raise NameError('缺少开发样本')
        try:
            qe = datasets['off']
        except:
            raise NameError('缺少oot样本')
        try:
            qe = datasets['dev'][dep]
        except:
            raise NameError('标签不在开发样本中')
        try:
            qe = datasets['off'][dep]
        except:
            raise NameError('标签不在oot样本中')

        var_names = [i for i in var_names if i != dep and i != uid]
        
        datasets['dev'] = datasets['dev'].fillna(fill_nan_with)
        datasets['off'] = datasets['off'].fillna(fill_nan_with)
        
        self.datasets = datasets  # 数据集字典{"dev":dev,"off":off}
        self.uid = uid  # 用户标识
        self.dep = dep  # 分类标签
        self.var_names = var_names  # 变量名列表
        self.params = params  # lightgbm输入参数
        self.early_stopping_rounds = early_stopping_rounds  # lightgbm早停轮次
        self.min_data = max(int(len(self.datasets.get("dev", "")) * 0.01), 50)
    
    def train(self,select_feature=False, single_delete=True, target=False,select_type='shap',
                  imp_threhold=0, corr_threhold=0.7, psi_threhold=0.1, params_weight=0.2):
            """
            :param no_select: boolean， 是否开启变量筛选
            :param single_delete: boolean， # 是否逐步特征删除
            :param target: string， 自动调参方法["weight","offks"," avg"],else尝试减少模型过拟合倾向
                    "offks": offks最大化;
                    "minus": 1-abs(devks-offks) 最大化;
                    "avg": (devks+offks)/2  最大化
                    "weight": offks + abs(offks - devks) * 0.2 最大化
                    False: 不进行调参
            :param imp_threhold: float，特征重要性阈值，"shap" 或 "feature_importance"都需要大于等于该阈值
            :param corr_threhold: float, 相关性阈值
            :param psi_threhold: float, 单变量PSI阈值上限
            :param params_weight: float, weight调参方法的权重
            :return: lighrgbm.Booster
            """
            
            print('开始自动建模...')
            print('-'*50)
            
            # 获取变量名和数据集
            var_names = self.var_names.copy()
            
            dev_data = self.datasets.get("dev", "")[var_names + [self.dep]]
            off_data = self.datasets.get("off", "")[var_names + [self.dep]]
            
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
                "num_boost_round": self.params.get("num_boost_round", 100)}
    
            # base model
            model = lgb.train(params=params, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False,
                              train_set=lgb.Dataset(dev_data[var_names], dev_data[self.dep]),
                              valid_sets=lgb.Dataset(off_data[var_names], off_data[self.dep]))
    
            # 变量筛选
            if select_feature:
                var_names = feature_select(self.datasets, model, var_names, self.dep, select_type, imp_threhold, corr_threhold,psi_threhold)
                
            # 自动调参
            if target:
                params = auto_choose_params(self.datasets, var_names, self.dep, self.min_data, 
                                            params_weight, early_stopping_rounds=self.early_stopping_rounds, params=params, target=target)
                print('-' * 50)
    
            # 逐个特征删除
            if single_delete:
                _, var_names = auto_delete_vars(self.datasets, var_names, self.dep, self.min_data,
                                                self.early_stopping_rounds, params=params)
                print("-" * 50)
    
            # final model
            model = lgb.train(params=params, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False,
                              train_set=lgb.Dataset(dev_data[var_names], dev_data[self.dep]),
                              valid_sets=lgb.Dataset(off_data[var_names], off_data[self.dep]))
            
            # 计算KS和PSI
            devks = sloveKS(model, dev_data[var_names], dev_data[self.dep])
            offks = sloveKS(model, off_data[var_names], off_data[self.dep])
            offpsi = slovePSI(model, dev_data[var_names], off_data[var_names])
            dic = {"devks": devks, "offks": offks, "offpsi": offpsi}
            print("KS & PSI: ", dic)
            print("-" * 50)
            print("AutoML建模完成")
    
            return model,var_names
