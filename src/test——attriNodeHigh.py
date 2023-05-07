import numpy as np
import networkx as nx
import algorithms.node2vec_attrHigh_new1 as node2vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy import spatial

def cos_similarity(v1, v2):
    # 余弦相似度
    return 1 - spatial.distance.cosine(v1, v2)

def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input_file, nodetype=args.node_type, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input_file, nodetype=args.node_type, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    # 给边打上属性距离
    for edge in G.edges():
        G[edge[0]][edge[1]]['distance'] = -1  # 初始化为-1

    if not args.directed:
        G = G.to_undirected()

    if args.attributed:  # 读取节点feature
        file_path = args.node_feature_file
        file = open(file_path, "rb")
        lines_ = (line if isinstance(line, str) else line.decode("utf-8") for line in file)
        feat_dict = node2vec.parse_feat(lines=lines_, comments="#", delimiter=None, node_type=args.node_type)
    else:
        feat_dict = {}
    # 可视化
    # G = nx.les_miserables_graph()  # !!!!
    # plt.figure(figsize=(15, 14))
    # pos = nx.spring_layout(G, seed=10)
    # nx.draw(G, with_labels=True)
    # plt.show()
    # print("len(G) = ", len(G))
    # print("G.nodes: ", G.nodes)
    # print("G.edges: ", G.edges)

    return G, feat_dict


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    # walks_ = [map(str, walk) for walk in walks]
    # 错误原因： 仅把walk转换成了str, 内部的词（node）没有转换，打印出来的类型为 map object
    # walks_ = list(map(str, walk) for walk in walks)
    # a = list(map(str, 'python'))
    # print("a: ", a)
    # walks_ = []
    # for walk in walks:
    #     walk_string = map(str, walk)
    #     walks_.append(walk_string)
    # print("walks_[0]: ", walks_[0])
    # print(walks)
    # print(len(walks))
    # model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size,
    #                  min_count=1, batch_words=4, sg=1,
    #                  workers=args.workers, epochs=args.iter)
    # model.wv.save_word2vec_format(args.output)

    # 将node的类型int转化为string
    walk_str = []
    for walk in walks:
        tmp = []
        for node in walk:
            tmp.append(str(node))
        walk_str.append(tmp)

    # 调用 gensim 包运行 word2vec
    model = Word2Vec(walk_str, vector_size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1,
                     workers=args.workers, epochs=args.iter)
    # 导出embedding文件
    model.wv.save_word2vec_format(args.output_file)

    return model

def clustering(model):
    X = model.wv.vectors  # 词向量
    cluster_model = SpectralClustering(n_clusters=args.n_clusters).fit(X)
    cluster_labels = cluster_model.labels_
    # cluster_labels = KMeans(n_clusters=args.n_clusters, random_state=9).fit(X).labels_
    # print(cluster_labels)
    # with open(args.output_label_file, 'w') as f:
    #     for label in cluster_labels:
    #         f.write(str(label)+"\n")
    return cluster_labels, cluster_model

def visualize_cluster(G, model, cluster_labels):
    colors = []
    nodes = list(G.nodes)
    for node in nodes:  # 按 networkx 的顺序遍历每个节点
        idx = model.wv.key_to_index[str(node)]  # 获取这个节点在 embedding 中的索引号
        # colors.append(cluster_labels[idx])  # 获取这个节点的聚类结果
        label = cluster_labels[idx]
        if label == 0:
            colors.append("#e8c102")
        elif label == 1:
            colors.append("#2c9b49")
        elif label == 2:
            colors.append("#c096f3")
        elif label == 3:
            colors.append("#359bfd")
        elif label == 4:
            colors.append("#ba1246")
        elif label == 5:
            colors.append("#EE7942")
    # pos = nx.spring_layout(G, seed=10)
    # pos = nx.spectral_layout(G)
    # nx.draw(G, pos, node_color=colors, with_labels=True)
    # plt.figure(figsize=(10, 10))
    # nx.draw(G, node_color=colors, node_size=750, with_labels=True)
    plt.figure(figsize=(25, 20))
    nx.draw(G, node_color=colors, node_size=270, with_labels=False)
    plt.show()

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G, feat_dict = read_graph()
    # nx_G = nx.les_miserables_graph().to_undirected()
    # b = nx_G.has_edge(10, 0)
    # edges =  nx_G.edges()
    G = node2vec.Graph(nx_G, args.directed, args.attributed, args.p, args.q, args.gamma, args.beta, feat_dict)
    # 测试 -
    # 节点概率alias sampling和归一化
    # is_directed = args.directed
    # alias_nodes = {}
    # alias_edges = {}
    # triads = {}
    # for node in nx_G.nodes():
    #     unnormalized_probs = [nx_G[node][nbr]['weight'] for nbr in sorted(nx_G.neighbors(node))]
    #     norm_const = sum(unnormalized_probs)
    #     normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
    #     alias_nodes[node] = node2vec.alias_setup(normalized_probs)
        # 信息展示
        # if node == 'Napoleon':
        #     print('Napoleon节点')
        #     print(unnormalized_probs)
        #     print(norm_const)
        #     print(normalized_probs)
        #     print(alias_nodes[node])
    # 边概率alias sampling和归一化
    # if is_directed:
    #     for edge in nx_G.edges():
    #         alias_edges[edge] = G.get_alias_edge(edge[0], edge[1])
    # else:
    #     for edge in nx_G.edges():
    #         alias_edges[edge] = G.get_alias_edge(edge[0], edge[1])
    #         alias_edges[(edge[1], edge[0])] = G.get_alias_edge(edge[1], edge[0])
    # 每个节点的 J表 和 q表 （不考虑p、q值）
    # print("alias_nodes: ", alias_nodes)
    # 二阶随机游走（考虑p、q值）
    # print(alias_edges)
    # 生成一条随机游走序列
    # G.preprocess_transition_probs()
    # walk = G.node2vec_walk(7, 'OldMan')
    # print(walk)

    G.preprocess_distance()
    G.preprocess_transition_probs()
    # print(G.alias_nodes)
    walks = G.simulate_walks(args.num_walks, args.walk_length)  # 采样得到所有随机游走序列
    # print("walks[0]: ", walks[0])
    # print(len(walks))
    # 利用Word2Vector计算embeddings
    model = learn_embeddings(walks)
    # print(model.wv.get_vector('OldMan').shape)
    # print(model.wv.get_vector('OldMan'))  # 词向量
    # print(model.wv.similarity('OldMan', 'Napoleon'))  # 节点对的相似度！！！！这里不一样
    # 找到最相似的节点
    # print(model.wv.most_similar('OldMan'))

    # 聚类
    cluster_labels, cluster_model = clustering(model)
    # 计算silhouette_score
    silhouette = silhouette_score(model.wv.vectors, cluster_labels, metric='euclidean')
    print("Silhouette = ", silhouette)

    # print(cluster_labels)
    # 聚类可视化
    visualize_cluster(nx_G, model, cluster_labels)

    # PCA 降维  # 测试
    X= model.wv.vectors
    pca = PCA(n_components=2)
    embed_2d = pca.fit_transform(X)
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.gca()
    # ax.set_facecolor('white')
    # ax.plot(X[:, 0], X[:, 1], '^', markersize=7, alpha=1, color="red")
    # nodes = list(model.wv.index_to_key)
    # for i, node in enumerate(nodes):
    #     plt.annotate(node, xy=(embed_2d[i,0], embed_2d[i, 1]))
    # plt.savefig('../emb/twitter/629863/629863.jpg')
    # plt.show()

    # TSNE 降维
    # X = model.wv.vectors
    # tsne = TSNE(n_components=2)
    # embed_2d = tsne.fit_transform(X)
    # 谱嵌入
    # X = model.wv.vectors
    # se = SpectralEmbedding(n_components=8)
    # embed_3d = se.fit_transform(X)
    # pca = PCA(n_components=2)
    # embed_2d = pca.fit_transform(embed_3d)

    nodes = list(model.wv.index_to_key)
    clist = []
    special = []
    # special = ['0','1','2','3','4','5','6','7','8','9','10'] # 5common, noise 1
    # special = ['9', '30', '28', '31', '33', '32', '27', '25', '24', '23',
    #            '29', '26', '15', '14', '20', '22', '18'] # karate
    # noise 2
    # special = ['1','2','3','4','12','13','14','15','16','19','20','21','22']
    # center = ['0','5','6','11','17','18','27']
    # leaf = ['7','8','9','10','23','24','25','26']

    # special = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59']
    # special = ['21', '19']
    # special = ['0','1','2','3','4','5','6','7','8','9',
    #            '22','23','24','25','26','27','28','29','30','31']
    # special = ['1','2','7','8','12','13','19','20','23','24']
    for i, node in enumerate(nodes):
        # plt.annotate(node, xy=(embed_2d[i,0], embed_2d[i, 1]))  # 散点标签
        if(node in special):
            clist.append('darkorange')
        # elif (node in center):
        #     clist.append('yellow')
        # elif (node in leaf):
        #     clist.append('aquamarine')
        else:
            clist.append('indigo')
    n = 0
    # special_num = len(special)
    # sum = 0
    # while n < special_num-1:
    #     m = n + 1
    #     v1 = model.wv.get_vector(special[n])
    #     v2 = model.wv.get_vector(special[m])
    #     sum += cos_similarity(v1, v2)
    #     n += 1
    # print("Special Compactness =", sum)
    # print(clist)
    plt.scatter(embed_2d[:, 0], embed_2d[:, 1], s=20, c=clist)
    plt.show()

    # embedding降维
    # rawNodeVec = []
    # node2ind = {}
    # for i, w in enumerate(model.wv.index_to_key):
    #     rawNodeVec.append(model.wv[w])  # 词向量
    #     node2ind[w] = i  # {词语：序号}
    # rawNodeVec = np.array(rawNodeVec)
    # X_reduced = PCA(n_components=2).fit_transform(rawNodeVec)
    # print("shape before", rawNodeVec.shape)
    # print("shape after", X_reduced.shape)
    # # # 绘图
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.gca()
    # ax.set_facecolor('white')
    # ax.plot(X_reduced[:, 0], X_reduced[:, 1], '^', markersize=7, alpha=1, color="red")
    # plt.savefig('../figure/test.jpg')
    # plt.show()


class ARGS:
    def __init__(self):
        input = "../graph/barbell/barbell.edgelist"
        output = "../emb/barbell.emb"
        node_type = int
        attributed = False
        dimensions = 16
        p = 1
        q = 2
        walk_length = 10
        num_walks = 100
        window_size = 3
        iter = 1
        workers = 4
        weighted = False
        directed = False
        n_clusters = 4
        gamma = 0  # gamma for 'gamma'-loosely
        beta = 0


if __name__ == "__main__":
    # args = ARGS()
    # args.input = "../graph/twitter/629863.edges"
    # args.output = "../emb/twitter/629863/629863.emb"
    # args.node_feature_file = "../graph/twitter/629863.feat"
    # args.node_type = int
    # args.dimensions = 16
    # args.walk_length = 10
    # args.num_walks = 20
    # args.window_size = 3
    # args.iter = 1
    # args.workers = 4
    # args.p = 1
    # args.q = 2
    # args.weighted = False
    # args.directed = False
    # args.attributed = True
    # args.n_clusters = 3
    # args.gamma = 0
    # args.beta = 2
    # main(args)

    args = ARGS()
    # cit-HepPh
    args.input_file = "../graph/cit-HepPh/cit-HepPh.edgelist"
    args.output_file = "../emb/cit-HepPh/testA.emb"
    args.node_feature_file = "../graph/cit-HepPh/cit-HepPh-dates.feat"
    # karate
    # args.input_file = "../graph/karate/karate.edgelist"
    # args.output_file = "../emb/karate/testA.emb"
    # args.node_feature_file = "../graph/karate/karate_club.feat"
    # args.output_label_file = "../emb/karate/test_attr_label.txt"
    # 5-common
    # args.input_file = "../graph/5common/5common.edgelist"
    # args.output_file = "../emb/5common/n2v_attributed/group1b/beta0"
    # args.node_feature_file = "../graph/5common/5common_Noise1.feat"
    #twitter
    # args.input_file = "../graph/twitter/629863.edges"
    # args.output_file = "../emb/twitter/test"
    # args.node_feature_file = "../graph/twitter/629863.feat"
    args.node_type = int
    args.dimensions = 32
    args.walk_length = 8
    args.num_walks = 20
    args.window_size = 5
    args.iter = 1
    args.workers = 4
    args.p = 2
    args.q = 0.05
    args.weighted = False
    args.directed = True
    args.attributed = True
    args.n_clusters = 6
    args.gamma = 0
    args.beta = 8
    main(args)