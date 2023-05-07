import gensim
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# tencent 预训练的词向量文件路径

vec_path = "../emb/twitter/629863/629863.emb"
# 加载词向量文件
wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
X = wv_from_text.vectors
# X= model.wv.vectors
pca = PCA(n_components=2)
embed_2d = pca.fit_transform(X)
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X[:, 0], X[:, 1], '^', markersize=7, alpha=1, color="red")
nodes = list(wv_from_text.index_to_key)
for i, node in enumerate(nodes):
    plt.annotate(node, xy=(embed_2d[i,0], embed_2d[i, 1]))
# plt.savefig('../emb/twitter/629863/629863.jpg')
plt.show()
