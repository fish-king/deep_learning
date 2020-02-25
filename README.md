# deep_learning
Documenting the Deep Learning Process
基础逻辑架构模型(2隐藏层4：1)：logistic.py

改善优化模型：
偏差(bias):模型的预测值与实际值之间的偏离关系----模型欠拟合
方差(variance):训练集准确率和验证集准确率偏离关系----模型过拟合

高偏差：扩大网络结构、延长训练时间、更换模型结构
高方差：增大训练集、正则化、更换模型结构

正则化：加在costfunction后
l2正则化(权重缩减):λ/2m ‖w‖^2      
l1正则化:λ/2m ‖w‖        结果中w稀疏含极多0，但并非对模型压缩

λ↑ w↓ (相当于降低很多隐藏单元权重) z↑ 模型接近线性模型，过拟合度降低

dropout:随机神经元消失本质简化网络

inverted dropout:
d3 = np.random.rand(a3.shape[0],a3.shape[1]) < keep-prop
a3 = np.multply(a3,d3)
a3 /= keep-prop #修正不改变z4期望值
