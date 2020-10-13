# Decision tree

## Programming ISOMAP, LLE 对流形的降维

考虑如下的问题并实现 ISOMAP，LLE 等降维方法：

2.1 在三维空间中产生 “Z” 形状的流形，使用 ISOMAP 方法降维并作图，给出数据的三维分布图和最佳参数下的降维效果图。

2.2 在三维空间中产生 “3” 形状的流形，使用 LLE 方法降维并作图，给出数据的三维分布图和最佳参数下的降维效果图。

**注意**：数据在产生过程中可不必严格保证形状，大致符合要求即可。不用在数据的产生上花费过多时间。

## Progamming Decision Tree
实现 **决策树** 算法，并且在 $Sogou Corpus$ 数据集上测试它的效果（数据集详情见readme）. 

要求：
- 不能调用已有的机器学习包
- 将数据**随机**分为**3:1:1**的三份，分别为训练集，交叉验证集，测试集。请在训练集上训练，交叉验证集上选择超参数，用选出的最好模型在测试集上给出测试结果。因此，在报告中需要说明算法的超参数有哪些，在不同的超参数设置下训练集和交叉验证集的分类正确率，最好模型的超参数设置，以及最后的测试正确率。
- 请结构化代码，必须包含但不限于如下几个函数（请从代码中分离出来，有明确的这几个函数，函数参数可以有所变化）：
   - \textbf{main()}
     
   - \textbf{要求main函数在运行中，逐个测试不同的超参数，然后打印出每个超参数的设置，该设置下的训练、验证正确率（就是上面第二点中提到的要出现在报告中的结果）。}
   - \textbf{GenerateTree(args)} 生成树的总代码，\emph{args}为各种超参数，包括但不限于下面的\emph{thresh}，或者其他会影响树性能的超参数，自由发挥。
   - textbf{SplitNode(samplesUnderThisNode, thresh, \dots)}对当前节点进行分支，\emph{samplesUnderThisNode}是当前节点下的样本，\emph{thresh}是停止分支的阈值，停止分支的条件请在实验报告中说明。
   - \textbf{SelectFeature(samplesUnderThisNode, \dots)}对当前节点下的样本，选择待分特征。
   - \textbf{Impurity(samples)}给出样本\emph{samples}的不纯度，请在实验报告中说明采用的不纯度度量。
   - \textbf{Decision(GeneratedTree, XToBePredicted)}使用生成的树\emph{GenerateTree}，对样本\emph{XToBePredicted}进行预测。
   - textbf{Prune(GeneratedTree, CorssValidationDataset, \dots)}对生成好的树\emph{GeneratedTree}（已经经过 stopped splitting) 进行剪枝：考虑所有相邻的叶子节点，如果将他们消去可以增加验证集上的正确率，则减去两叶子节点，将他们的共同祖先作为新的叶子节点。或者实现其他的剪枝方法，如有，请在实验报告中说明。