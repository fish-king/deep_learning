# deep_learning
Documenting the Deep Learning Process
基础逻辑架构模型(2隐藏层4：1)：logistic.py

改善优化模型：
偏差(bias):模型的预测值与实际值之间的偏离关系----模型欠拟合
方差(variance):训练集准确率和验证集准确率偏离关系----模型过拟合

高偏差：扩大网络结构、延长训练时间、更换模型结构
高方差：增大训练集、正则化、更换模型结构

正则化：加在costfunction后
l2正则化(权重缩减):λ/2m ‖w‖₂²    
l1正则化:λ/2m ‖w‖₁        结果中w稀疏含极多0，但并非对模型压缩
‖w‖₁即1-范数：向量元素绝对值之和
‖w‖₂即2-范数：Euclid范数，即向量元素绝对值的平方和再开方

λ↑ w↓ (相当于降低很多隐藏单元权重) z↓ z范围小则在激活函数上接近线性，模型也贴近线性模型，过拟合度降低

dropout正则化:随机神经元消失本质简化网络

inverted dropout:
d3 = np.random.rand(a3.shape[0],a3.shape[1]) < keep-prop
a3 = np.multply(a3,d3)
a3 /= keep-prop #修正不改变z4期望值

early stoping:在验证集准确率与训练集准确率差别增大时提前结束训练
data argumentation：增大数据集，或者修改原图片扩充数据集

归一化:提升训练速度(要求数据近似高斯分布)
零均值化：
u = 1/m∑x(i)
x = x-u
标准化:
σ² = 1/m∑x(i)**2
x /= σ²

梯度爆炸和梯度消失：常见深层次网络W初始化经n层指数级对loss影响，会是梯度剧烈波动
解决方法:
初始化参数w=np.random.randn(shape)*np.sqrt(1/n^[l-1])
n^[l-1]为上一次神经元个数
ps:对于ReLu激活函数，一般选取np.sqrt(2/n^[l-1])

梯度检测实际上用双边误差公式近似求出 f'(x) = lim(ε->0) (f(x+ε)-f(x-ε))/2ε (高数中导数及无穷小应用)
dθapprox ≈ dθdropout
‖dθapprox - dθ‖₂/(‖dθapprox‖₂+‖dθ‖₂)与ε数量级接近，则认为梯度函数优秀for debug 不与dropout合用
