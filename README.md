# 计算机视觉作业1：Fashion-MNIST 三层MLP分类器

## 一、项目简介
本项目基于三层全连接神经网络（MLP）完成Fashion-MNIST服装10分类任务。
手动实现前向传播、反向传播、梯度下降、L2正则化等功能。

## 二、环境依赖
python >= 3.7
numpy
matplotlib
scikit-learn
torchvision

安装命令：
pip install numpy matplotlib scikit-learn torchvision

## 三、文件清单
main.py：主程序（数据加载、模型定义、训练、测试、可视化）
log.txt：超参数网格搜索 + 完整训练日志
error_log.txt：测试集分类错误样本记录
tmp.npz：训练过程临时权重文件
final_best_model.npz：最优模型权重文件
train_curve.png：训练/验证损失与准确率曲线
confusion_matrix.png：分类混淆矩阵
all_neurons.png：第一层神经元权重可视化
error_analysis.png：测试集错例可视化

## 四、运行方法
1. 确保 datasets 文件夹与 main.py 在同一级目录
2. 直接运行主程序：
python main.py

程序自动执行流程：
数据集加载、超参数网格搜索、模型训练、保存最优权重、测试评估、图表输出

## 五、实验结果
最优超参数：hidden_dim=256，learning_rate=0.1，lambda_l2=0.0001
测试集准确率：88.33%

## 六、模型权重下载
