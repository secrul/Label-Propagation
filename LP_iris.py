import copy
import time
import numpy as np
import math
from sklearn import datasets
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def navie_knn(dataSet, query, k):
    #找出样本点的k个最近邻居，然后图只与这些最近邻数据点相连，稀疏化图结构
    numSamples = dataSet.shape[0]

    ## step 1: 计算节点之间的欧式距离
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row

    ## step 2: 距离排序，找出最近的k个邻居
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)

    return sortedDistIndices[0:k]
#
#
# 构建图
def buildGraph(MatX, kernel_type, rbf_sigma=None, knn_num_neighbors=None, distance_r=None):
    # 目的是利用节点之间的相似性构建图，至于怎么衡量相似性，可以用欧式距离，余弦夹角，knn等
    num_samples = MatX.shape[0] #数据的总数目
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32) #转移矩阵，就是图结构
    if kernel_type == 'rbf': #rbf方法
        if rbf_sigma == None:
            raise ValueError('You should input a sigma of rbf kernel!')
        for i in range(num_samples):#分别计算两个节点之间的rbf距离
            row_sum = 0.0
            for j in range(num_samples):
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i][j] = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] //= row_sum
    elif kernel_type == 'knn': #只有k个邻居才有权重，权重值为1/k
        if knn_num_neighbors == None:
            raise ValueError('You should input a k of knn kernel!')
        for i in range(num_samples):
            k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError('Not support kernel type! You can use knn or rbf!')

    return affinity_matrix


# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='rbf', rbf_sigma=1.5, \
                     knn_num_neighbors=10, distance_r = 1.0, max_iter=500, tol=1e-4):
    """

    :param Mat_Label: 训练数据
    :param Mat_Unlabel: 测试数据
    :param labels: 训练标签
    :param kernel_type: 构建图的方法，rbf或者knn
    :param rbf_sigma: rbf的超参
    :param knn_num_neighbors: knn的超参
    :param max_iter: 最大迭代次数
    :param tol: 停止迭代的阈值，当更新的值小于tol停止迭代
    :return: 预测的标签
    """
    import time
    t1 = time.time()
    num_label_samples = Mat_Label.shape[0] #有标签数据的个数
    num_unlabel_samples = Mat_Unlabel.shape[0] #无标签数据个数
    num_samples = num_label_samples + num_unlabel_samples #数据总个数
    labels_list = np.unique(labels)
    num_classes = len(labels_list) #label类别的数目
    MatX = np.vstack((Mat_Label, Mat_Unlabel)) #有标签数据和无标签数据合并
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)#gt,每次迭代来矫正已经标注数据
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0
    label_function = np.zeros((num_samples, num_classes), np.float32) #记录类别转移矩阵
    label_function[0: num_label_samples] = clamp_data_label
    label_function[num_label_samples: num_samples] = -1

    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors, distance_r)#图，数据之间的相关性
    t2 = time.time()
    print('bulid graph', t2-t1) #构建图的时间，这个随着数据个数增长，O(n^2)
    # start to propagation
    iter = 0
    pre_label_function = np.zeros((num_samples, num_classes), np.float32) #类别预测值，每次变换矩阵后的结果
    changed = np.abs(pre_label_function - label_function).sum() #每次迭代类别矩阵变换大小
    while iter < max_iter and changed > tol:#当迭代次数大于设定，或者每次的变化值小于阈值，停止迭代
        if iter % 1 == 0:
            print("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))
        pre_label_function = label_function
        iter += 1
        # propagation
        label_function = np.dot(affinity_matrix, label_function) #使用转移矩阵，迭代
        # clamp
        label_function[0: num_label_samples] = clamp_data_label#有标签的数据使用真实标签替换
        # check converge
        changed = np.abs(pre_label_function - label_function).sum()
        t3 = time.time()
        print('iter time', t3-t2)
        t2 = t3
        # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):#
        unlabel_data_labels[i] = np.argmax(label_function[i + num_label_samples])
    return unlabel_data_labels


def scatter_diagram(data, label):
    '''
    #画路径图
    输入：nodes-节点坐标；
    输出：散点图
    '''
    for i in range(len(data)):
            if label[i] == 0:
                plt.scatter(data[i][0], data[i][1], alpha=0.8, c='r')
            if label[i] == 1:
                plt.scatter(data[i][0], data[i][1], alpha=0.8, c='g')
            if label[i] == 2:
                plt.scatter(data[i][0], data[i][1], alpha=0.8, c='b')
            if label[i] == -1:
                plt.scatter(data[i][0], data[i][1], alpha=0.8, c='c')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":

    """
    dev.csv 36572个数据， id, 600维向量 预处理数据，归一化等操作
    development_data_labels.csv dev_id和speaker_id映射，4958个speaker_id
    :return:
    """
    import time
    t1 = time.time()
    #注意 num1 > num2
    Num1 = 21 #去除低于Num1数据个数的类别
    Num2 = 20 #每个类别Num2个数据作为训练，其余测试
    sigma = 0.2
    k = 10
    r = 1.7
    #load dev_id
    pca = PCA(n_components=2)
    iris = datasets.load_iris()
    rate = 0.7
    label = np.copy(iris.target)

    unlabel_index = np.random.rand(len(label))

    unlabel_index = unlabel_index < rate

    label_index = [~x for x in unlabel_index]

    train_data = iris.data[label_index]
    train_label = label[label_index]
    test_data = iris.data[unlabel_index]
    test_label = label[unlabel_index]
    # print(iris.data.shape)
    # assert  1 > 9
    pca.fit(iris.data)
    plotx = pca.transform(iris.data)
    scatter_diagram(plotx, label)
    # assert 2 > 9
    tmp_label = copy.deepcopy(label)
    tmp_label[unlabel_index] = -1
    scatter_diagram(plotx, tmp_label)
    print("无标注数据：", test_label.shape)

    # unlabel_data_labels = labelPropagation(train_data, test_data, train_label, kernel_type = 'rbf', rbf_sigma = sigma)
    Pre_labels = labelPropagation(train_data, test_data, train_label, kernel_type='knn', knn_num_neighbors=k,
                                           max_iter=400)
    t2 = time.time()
    ultradata = np.concatenate((train_data, test_data),axis=0)
    ultralabel = np.concatenate((train_label, Pre_labels),axis=0)
    ultradata = pca.transform(ultradata)
    scatter_diagram(ultradata, ultralabel)
    print(t2 - t1)

    from sklearn.metrics import accuracy_score, recall_score, f1_score

    print('ACC,', accuracy_score(test_label, Pre_labels))


