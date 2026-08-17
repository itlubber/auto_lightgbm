# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:41:55 2020

@author: meizihang
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import *

from hscredit.core.metrics.classification import ks
from hscredit.core.metrics.feature import iv_table
from hscredit.core.metrics.stability import psi

from .logger import logger


def solveIV(dev_data, var_names, dep, iv_only=True, cpu_cores=1):
    """
    调用 hscredit 计算IV（dataframe）
    """
    records = []

    for feature in var_names:
        try:
            table = iv_table(dev_data[dep], dev_data[feature])
            feature_iv = float(table["分档IV值"].sum()) if "分档IV值" in table.columns else np.nan
            strategy_degree = float(table["分档WOE值"].abs().sum()) if "分档WOE值" in table.columns else np.nan
        except Exception:
            feature_iv = np.nan
            strategy_degree = np.nan

        records.append({"特征": feature, "IV": feature_iv, "策略度": strategy_degree})

    IV = pd.DataFrame(records).set_index("特征")
    return IV


def sloveKS(model, X, Y):
    """
    计算dev和oot上的KS值
    """
    return ks(Y, model.predict(X))


def slovePSI(model, dev_x, val_x):
    """
    计算oot相对于dev的PSI
    """
    return psi(model.predict(dev_x), model.predict(val_x))


def confusion_matrix(y, pred):
    # 产生混淆矩阵的四个指标
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    # 产生衍生指标
    fpr = fp / (fp + tn)  # 假真率／特异度
    tpr = tp / (tp + fn)  # 灵敏度／召回率
    depth = (tp + fp) / (tn + fp + fn + tp)  # Rate of positive predictions.
    ppv = tp / (tp + fp)  # 精准率
    lift = ppv / ((tp + fn) / (tn + fp + fn + tp))  # 提升度
    afdr = fp / tp  # (虚报／命中)／好账户误判率
    return lift


def normall_evl(valid_y, y_pred):
    """
    单类计算各种评价指标
    """
    dct = {}
    dct["分类准确率为"] = accuracy_score(valid_y, y_pred)
    dct["宏平均准确率"] = precision_score(valid_y, y_pred, average="macro")
    dct["微平均准确率"] = precision_score(valid_y, y_pred, average="micro")

    dct["宏平均召回率为"] = recall_score(valid_y, y_pred, average="macro")
    dct["微平均召回率为"] = recall_score(valid_y, y_pred, average="micro")

    dct["宏平均f1值为"] = f1_score(valid_y, y_pred, average="macro")
    dct["微平均f1值为"] = f1_score(valid_y, y_pred, average="micro")
    dct["lift值为"] = confusion_matrix(valid_y, y_pred)
    return dct


def evl_all(df, dep, pred_class):
    """
    多类分别计算评价指标
    """
    for i in set(df[pred_class]):
        y_label = df[dep]
        logger.info(f"{i}\t{normall_evl(y_label, pred_class)}")
