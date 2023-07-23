## 概述

梅子行老师那里来的一个 `lightgbm` 建模的方法，配合 `scorecardpipeline` 一起使用非常方便，效果也非常可以，值得推荐


## 总共2个功能
1 lightgbm快速建模和调参，产出报告等一系列动作     


## lightgbm

### Part1.eda分析 [pre_analyze](docs/pre_analyze.rst)

注意：
1、如果开启了这个过程，数据量大的时候跑的会比较慢  
2、特征用"f1_"和"f2_"等前缀来标注是否为同一数据源，方便评价数据源对目标的贡献，适用于新数据源评估

### 训练与调参

1 train 所有数据  
~~2 con_shap 则输出所有特征shap值（去除）~~  
3 判断特征自动筛选  
3.1 shap 或 feature importance  
3.2 correction  
3.3 PSI  
3.4 single_delete（逐个筛除）  
4 自动调参  
~~5 输出模型 ks和psi~~

### 评价与模型报告

1 feature importance （与训练3.1相似）  
2 shap（与训练3.1相似）  
3 roc  
4 classfication report at max ks cut-off point(boxcox转换后分数取值也变化)  
5 ks table  
