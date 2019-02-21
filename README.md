# 个性化推荐系统
## 参考博客https://zhuanlan.zhihu.com/p/32078473
其模型架构是google开发的Wide & Deep模型，这里删除了一维的cnn，也就是文本卷积神经网络，使用keras重新编写
### 数据集为ml-1m，是6040个用户（21种职业，7个年龄段）对3952部电影（19种类型）的评分（满分为5分），在模型测试中数据集划分改用sklearn.model_selection的train_test_split函数
### model.png 是模型架构图
### test.log 是模型构建过程中的日志
### 模型在测试集中的均方差为0.7850，平均绝对误差为0.6995，参考博客得均方差在1附近
