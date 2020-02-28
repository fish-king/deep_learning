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

mini-patch将全部数据分成一个个子集，每次对一个子集进行训练，如50m数据分为5000个1000大小子集，如此每次遍历全部数据后进行5000次梯度下降，相比原来一次梯度下降速度提升
mini-patch = 1 为随机梯度下降

指数加权平均:vt = βv(t-1) + (1-β)θt
初始时会产生偏差,因此用vt/(1-β^t)来降低偏差
之间均值的β倍加当前数值的(1-β)倍

在训练中应用:momentum梯度下降
vdw = βvdw + (1-β)dw
vdb = βvdb + (1-β)db
w = w - αvdw
b = b - αvdb
总体相当于平衡抖动，使梯度直线下降从而提升速度
ps:对于dw,db参数为1的momentum，应修改α使αvdw,αvdb倍率不变

β常设为0.9,Robust

RMSprop梯度下降
sdw = βsdw + (1-β)dw²
sdb = βsdb + (1-β)db²
w = w - α(dw/(√sdw+ε))
b = b - α(db/(√sdb+ε))

ε = 10^-8  也可省略，防止sdw，sdb为0情况
类似于momentum也是减缓非下降方向的抖动，不过momentum可以理解为矢量,RMSprop为标量，若sdb抖动过大，b降幅变小，降低抖动

Adam算法
vdw = 0,sdw = 0,vdb = 0,sdb = 0
vdw = β₁vdw + (1-β₁)dw
vdb = β₁vdb + (1-β₁)db
sdw = β₂sdw + (1-β₂)dw²
sdb = β₂sdb + (1-β₂)db²
#初值修正
vdw = vdw/(1-β₁^t)
vdb = vdb/(1-β₁^t)
sdw = sdw/(1-β₂^t)
sdw = sdw/(1-β₂^t)

w = w - α(vdw/(√sdw+ε))
b = b - α(vdb/(√sdb+ε))

常用取值：
学习率自取
β₁ = 0.9
β₂ = 0.999
ε = 10^-8  也可省略，影响不大

学习率衰减有多种方式，自己选择

鞍点，因维度高，故所有维度导数均为0的点几乎不存在，因此几乎无局部最优问题，但导数过小，训练速度会因此减缓，我们通过以上三种算法优化训练速度，相当于加大推动梯度下降方向的力
学习率衰减
α = 1/(1+decayrate*epohnum)  * α0
