# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:39:11 2020

@author: meizihang
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from scorecardpipeline import *
from automl import auto_lightgbm


X, y = make_classification(n_samples=1000,n_features=30,n_classes=2,random_state=328)
data = pd.DataFrame(X)
data.columns = [f"f{i}" for i in range(len(data.columns))]
data['target'] = y

target = "target"
seed = 2349
num_threads = 1
test_size = 0.25
imp_threhold = 1e-4
corr_threhold = 0.4
early_stopping_rounds = 5


dev, oot = train_test_split(data, test_size=test_size, random_state=seed, stratify=data[target])

lgb_base = auto_lightgbm({"dev": dev, "oot": oot}, params={'num_threads': num_threads}, early_stopping_rounds=early_stopping_rounds)
model, new_var_names = lgb_base.train(
                                        select_feature=True, # 特征筛选
                                        select_type='shap',  # 特征筛选指标
                                        single_delete=True,  # 逐个特征删除
                                        imp_threhold=0,      # 特征筛选指标阈值
                                        corr_threhold=0.7,   # 相关系数阈值
                                        psi_threhold=0.2,    # PSI阈值
                                        target='weight',     # 参数搜索目标函数
                                        params_weight=0.2    # weight目标函数权重
                                    )

logistic = ITLubberLogisticRegression(target=target, class_weight={1: 0.9, 0: 0.1}, C=10, max_iter=50)
logistic.fit(data[new_var_names + [target]])
summary = logistic.summary()

print(summary)
