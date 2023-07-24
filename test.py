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

from automl.aml_main import auto_lightgbm


X, y = make_classification(n_samples=1000,n_features=30,n_classes=2,random_state=328)
data = pd.DataFrame(X)
data['bad_ind'] = y
data['imei'] = [i for i in range(len(data))]
data.columns = ['f0_radius','f0_texture','f0_perimeter','f0_area','f0_smoothness',
                'f0_compactness','f0_concavity','f0_concave_points','f0_symmetry',
                'f0_fractal_dimension','f1_radius_error','f1_texture_error','f1_perimeter_error',
                'f2_area_error','f2_smoothness_error','f2_compactness_error','f2_concavity_error',
                'f2_concave_points_error','f2_symmetry_error','f2_fractal_dimension_error',
                'f3_radius','f3_texture','f3_perimeter','f3_area','f3_smoothness',
                'f3_compactness','f3_concavity','f3_concave_points','f3_symmetry',
                'f3_fractal_dimension','bad_ind','imei']

dev, off = train_test_split(data, test_size=0.3, random_state=328)

uid, dep = "imei", "bad_ind"
var_names = list(data.columns)
var_names.remove(dep)

datasets = {"dev": dev, "off": off}

lgb_base = auto_lightgbm(datasets, dep, var_names, uid, params={'num_threads': 1}, early_stopping_rounds=10)

#训练模型
model,new_var_names = lgb_base.train(select_feature=True, #特征筛选
                                     select_type='shap',  #特征筛选指标
                                     single_delete=True,  #逐个特征删除
                                     imp_threhold=0,      #特征筛选指标阈值
                                     corr_threhold=0.7,   #相关系数阈值
                                     psi_threhold=0.2,    #PSI阈值
                                     target='weight',     #参数搜索目标函数
                                     params_weight=0.2   #weight目标函数权重
                                     )
