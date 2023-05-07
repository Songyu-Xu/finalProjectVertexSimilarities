import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import seaborn as sns


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    print("Read graph......")
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    # G.add_node(10)
    # 可视化
    # plt.figure(figsize=(15, 14))
    # pos = nx.spring_layout(G, seed=10)
    # nx.draw(G, pos, with_labels=True)
    # plt.show()
    # print("len(G) = ", len(G))
    # print("G.nodes: ", G.nodes)
    # print("G.edges: ", G.edges)

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    # 将node的类型转化为string
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
    model.wv.save_word2vec_format(args.output)

    return model

def clustering_kmeans(model):
    X = model.wv.vectors  # 词向量
    cluster_labels = KMeans(n_clusters=args.n_clusters, random_state=9).fit(X).labels_
    # print(cluster_labels)
    return cluster_labels

def visualize_cluster(G, model, cluster_labels):
    colors = []
    nodes = list(G.nodes)
    for node in nodes:  # 按 networkx 的顺序遍历每个节点
        idx = model.wv.key_to_index[str(node)]  # 获取这个节点在 embedding 中的索引号
        colors.append(cluster_labels[idx])  # 获取这个节点的聚类结果
    plt.figure(figsize=(15, 14))
    pos = nx.spring_layout(G, seed=10)
    # pos = nx.spectral_layout(G)
    nx.draw(G, pos, node_color=colors, with_labels=True)
    plt.show()

def run_node2vec(args):
    # nx_G = read_graph()
    nx_G = nx.les_miserables_graph().to_undirected()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)  # 采样得到所有随机游走序列
    model = learn_embeddings(walks)  # 利用Word2Vector计算embeddings

    return model, nx_G

def calculate_compact(nx_G, model, cluster_labels, cluster_num):
    """
    计算每个簇中的平均最短路径长度，仅对连通图适用
    """
    print("Calculate compact matrix......")
    clusters = np.zeros((cluster_num, 2))  # cluster[number of nodes, sum of shortest length]
    nodes = list(nx_G.nodes)
    node_num = len(nodes)  # 图中节点数量
    # all_shortest_length = dict(nx.all_pairs_shortest_path_length(nx_G))  # 所有节点间的最短路径
    target_label = 0
    while target_label < cluster_num:  # 遍历所有簇别
        index_list = []  # 同一簇的顶点embedding(cluster_labels)索引号
        for index, label in enumerate(cluster_labels):
            if label == target_label:
                index_list.append(index)
        index_node_num = len(index_list)  # 这一簇中的顶点数量
        print("target_label = ", target_label, " index_node_num = ", index_node_num)
        nd1 = 0
        while nd1 < index_node_num - 1:  # 遍历这一簇中的节点对
            idx1 = index_list[nd1]  # id1在embeddings的索引号
            nd1_key = model.wv.index_to_key[idx1]  # nd1的key
            nd2 = nd1 + 1
            while nd2 < index_node_num:
                idx2 = index_list[nd2]
                nd2_key = model.wv.index_to_key[idx2]  # nd2的key
                # print(nd1_key, "-", nd2_key)
                try:
                    length = nx.dijkstra_path_length(nx_G, nd1_key, nd2_key) # 节点为字符串时
                    # length = nx.dijkstra_path_length(nx_G, int(nd1_key), int(nd2_key))  # 节点为数字时
                    clusters[target_label][1] += length
                    if length > 0:
                        a = clusters[target_label][0] + 1
                        clusters[target_label][0] = a  # 这个簇记录一条边
                    # print(nd1_key, "-", nd2_key, ": ", clusters[target_label][1])
                except:
                    # no paths between nd1-nd2
                    continue
                finally:
                    nd2 += 1
            nd1 += 1
        target_label += 1

    graph_average_compact = 0
    for cluster in clusters:
        if cluster[0] > 0:  # 这个簇中有边
            average = cluster[1] / cluster[0]
            # print(average)
        graph_average_compact += average

    return clusters, graph_average_compact

def draw_heatmap(args, compact_matrix, iterWL_max, iterNW_max):
    x_ticks = []
    y_ticks = []
    walk_length = args.walk_length_min
    num_walks = args.num_walks_min
    iterWL = 0
    iterNW = 0
    while iterWL <= iterWL_max:
        x_ticks.append(str(walk_length))
        walk_length = round(walk_length + args.walk_length_step, 1)
        iterWL += 1
    while iterNW <= iterNW_max:
        y_ticks.append(str(num_walks))
        num_walks = round(num_walks + args.num_walks_step, 1)
        iterNW += 1
    # print(x_ticks)
    # print(y_ticks)
    ax = sns.heatmap(compact_matrix, xticklabels=x_ticks, yticklabels=y_ticks)
    title = 'Heatmap for misles \n dim=' + str(args.dimensions) + ', p=' + str(args.p) + ', q=' + str(args.q) + \
            ', window_size=' + str(args.window_size) + ', cluster_num=' + str(args.n_clusters)
    ax.set_title(title)  # 图标题
    ax.set_xlabel('walk length')  # x轴标题
    ax.set_ylabel('number of walks')
    plt.show()
    figure = ax.get_figure()

def main(args):
    compact_matrix = np.zeros((round(((args.num_walks_max-args.num_walks_min)/args.num_walks_step) + 1),
                               round((args.walk_length_max-args.walk_length_min)/args.walk_length_step) + 1))
    print(compact_matrix.shape)
    walk_length = args.walk_length_min
    num_walks = args.num_walks_min
    iter_walk_length = 0
    iter_num_walks = 0
    iterWL_max = round((args.walk_length_max-args.walk_length_min)/args.walk_length_step)
    iterNW_max = round((args.num_walks_max-args.num_walks_min)/args.num_walks_step)
    # walk_length 与 num_walks 循环
    while iter_walk_length <= iterWL_max:
        args.walk_length = walk_length
        num_walks = args.num_walks_min
        iter_num_walks = 0
        while iter_num_walks <= iterNW_max:
            args.num_walks = num_walks
            print("\n******************************\n",
                  "num_walks = ", args.num_walks, ", walk_length = ", args.walk_length)
            num_walks = round(num_walks + args.num_walks_step, 1)
            # 运行node2vec
            model, nx_G = run_node2vec(args)
            # KMeans聚类
            cluster_labels = clustering_kmeans(model)
            # 计算compactness
            clusters_matrix, graph_average_compact = calculate_compact(nx_G, model, cluster_labels, args.n_clusters)
            # print(graph_average_compact)
            compact_matrix[iter_num_walks, iter_walk_length] = format(graph_average_compact, '.2f')
            iter_num_walks += 1
        walk_length = round(walk_length + args.walk_length_step, 1)
        iter_walk_length += 1

    # 画热力图
    draw_heatmap(args, compact_matrix, iterWL_max, iterNW_max)

    # 聚类可视化
    # visualize_cluster(nx_G, model, cluster_labels)



class ARGS:
    def __init__(self):
        input = "../graph/karate.edgelist"
        output = "../emb/karate.emb"
        dimensions = 32
        p = 1
        q = 2
        walk_length_min = 1
        walk_length_max = 20
        walk_length_step = 1
        walk_length = 1
        num_walks_min = 10
        num_walks_max = 100
        num_walks_step = 10
        num_walks = 10
        window_size = 3
        iter = 1
        workers = 4
        weighted = False
        directed = False
        n_clusters = 3


if __name__ == "__main__":
    args = ARGS()
    args.input = "../graph/simple.edgelist"
    args.output = "../emb/lesmis.emb"
    args.dimensions = 16
    args.walk_length_min = 1
    # args.walk_length_max = 20
    args.walk_length_max = 50
    args.walk_length_step = 1
    args.num_walks_min = 1
    # args.num_walks_max = 100
    args.num_walks_max = 50
    args.num_walks_step = 1
    args.window_size = 3
    args.iter = 1
    args.workers = 4
    args.p = 1
    args.q = 2
    args.weighted = False
    args.directed = False
    args.n_clusters = 3
    main(args)