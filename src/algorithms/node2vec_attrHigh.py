import numpy as np
import networkx as nx
import random

import sklearn.preprocessing


def parse_feat(lines, comments="#", delimiter=None, node_type=int):
	"""
	Parameters
	----------
	lines : list or iterator of strings
		Input data in edgelist format
	comments : string, optional
	   Marker for comment lines. Default is `'#'`. To specify that no character
	   should be treated as a comment, use ``comments=None``.
	delimiter : string, optional
	   Separator for node labels. Default is `None`, meaning any whitespace.
	node_type: type, optional
		Type of nodes. Default is 'int'.
	"""
	feat_dict = {}  # 字典中存储{nd1: [attr1, attr2], nd2: [attr1, attr2], ......}
	for line in lines:
		if comments is not None:
			p = line.find(comments)
			if p >= 0:
				line = line[:p]
			if not line:
				continue
		# split line, should have 2 or more
		s = line.strip().split(delimiter)
		# print(s)
		if len(s) < 2:
			continue
		node = node_type(s.pop(0))
		# print(type(node))
		feature = []
		for num in s:
			feat = float(num)
			feature.append(feat)
		# 多个feature的情况
		feat_dict[node] = feature
		d = s
	# TODO: 还需要判断feat里的节点数量与nx_G里的是否一致，节点是否存在于nx_G
	# print(feat_dict)
	return feat_dict

def calculate_threshold(G, gamma, v):
	"""
	Calculate the threshold given the centre vertex
	"""
	distances = []
	nbrs = sorted(G.neighbors(v))	# 提取所有的邻居节点
	if len(nbrs) > 0:
		for v_nbr in nbrs:
			distances.append(G[v][v_nbr]['distance'])
		print(distances)
		# 计算均值
		average = sum(distances) / len(distances)
		# 计算方差
		variance = np.var(distances)
	# 	# TODO: 计算threshold
		d = average + gamma * variance

	return d

class Graph():
	def __init__(self, nx_G, is_directed, is_attributed, p, q, gamma, beta, feat_dict):
		self.G = nx_G
		self.is_directed = is_directed
		self.is_attributed = is_attributed
		self.p = p
		self.q = q
		self.gamma = gamma
		self.feat_dict = feat_dict
		self.beta = beta
		self.kernel = [[]]

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print ('Simulate walks.......')
		for walk_iter in range(num_walks):
			# print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def isnotin_extended(self, src, dst):  # src:t, dst_nbr: x
		return 0

	def preprocess_distance(self, sigma=1):
		'''
        预计算节点之间属性的距离，并记录到边的属性‘distance’上
        '''

		G = self.G

		feat_dict = self.feat_dict
		k = len(feat_dict)  #节点数量
		# print(feat_dict)
		# j = len(feat_dict[0])	#节点所带的属性数量
		j = len(feat_dict.get(next(iter(feat_dict)))) #节点所带的属性数量
		# features, 第一位为node

		nd_array = np.array([], dtype="int")
		features = np.zeros((k, j))
		i = 0
		for node in feat_dict:
			# print("node=",node)
			# print(nd_array)
			nd_array = np.append(nd_array, node)
			# print(nd_array)
			node_feats = feat_dict[node]  # node特征
			j = 0
			for f in node_feats:
				features[i][j] = f
				j += 1
			# print(features[i])
			i += 1
		# print(features)
		# 归一化
		ss = sklearn.preprocessing.StandardScaler()
		Y = np.array(features[:])[:,:]
		X = ss.fit_transform(Y)

		# 存储归一化后的节点-属性对
		feat_dict_ss = dict((zip(nd_array, X)))
		# print(feat_dict_ss)

		# 计算节点对属性相似度
		for edge in G.edges():
			# print(features_ss[edge[0]])
			src_feat = feat_dict_ss.get(edge[0])
			dst_feat = feat_dict_ss.get(edge[1])
			distance = np.exp(-(np.linalg.norm(src_feat - dst_feat))**2)
			# print(distance)
			self.G[edge[0]][edge[1]]['distance'] = distance
			# print(G[edge[0]][edge[1]])
		# print(G.edges())
		return 0


	#  带 node feature的扩展
	def get_alias_edge_attributed(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q
		beta = self.beta
		feat_dict = self.feat_dict
		gamma = self.gamma

		unnormalized_probs = []

		# edges = G.edges()
		# b = G.has_edge(10,0)
		# feat_dict = self.feat_dict  # feature dictionary
		# t--v--x-
		dt = calculate_threshold(G, gamma, src)
		dv = calculate_threshold(G, gamma, dst)
		# 核心算法
		for dst_nbr in sorted(G.neighbors(dst)):
			dx = calculate_threshold(G, gamma, dst_nbr)
			if dst_nbr == src:  # 如果邻居节点为src则使用p
				# print("dst_nbr == src ", dst_nbr, "=", src)
				print("return node, p")
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			else:
				HasEdge = G.has_edge(dst_nbr, src)
				if HasEdge:  # 如果t-x有边则判断 in/out/noise
					if abs(G[dst_nbr][src]['distance']) > dx:  # |wx-wt| < dt, 判断wx相对dv
						if abs(G[dst_nbr][dst]['distance']) > dx:  # |wx-wv| > dv, noisy node
							print("noisy-node")
							unnormalized_probs.append(G[dst][dst_nbr]['weight'] * min(1, 1/q))
						else:  # |wx-wv| <= dv, out node
							print("out-node, 1/q")
							# TODO: Beta
							# alpha = 1/q + 1/q * (beta - abs(feat_dict[dst_nbr][0] - feat_dict[src][0])/dt)
							alpha = 1/q + (1-1/q) * abs((G[dst_nbr][src]['distance'])/dx)
							unnormalized_probs.append(G[dst][dst_nbr]['weight'] * alpha)
					else: # |wx-wt| <= dt, in node
						print("in-node, 1")
						unnormalized_probs.append(G[dst][dst_nbr]['weight'])  # in-node, alpha=1
				else:
					print("no edge between, 1/q")
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		# 归一化各条边的转移权重
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		# 执行 Alias Sampling
		return alias_setup(normalized_probs)

	# 普通node2vec
	def get_alias_edge(self, src, dst):  # src(t): edge[0], dst(v): edge[1]
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		# 核心算法
		for dst_nbr in sorted(G.neighbors(dst)):  # 遍历v的所有邻居
			if dst_nbr == src:  # 返回前一节点 d_tx = 0
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):  # d_tx = 1
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:  # d_tx = 2
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		# 归一化各条边的转移权重
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		# 执行 Alias Sampling
		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed
		is_attributed = self.is_attributed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			if is_attributed:
				for edge in G.edges():
					alias_edges[edge] = self.get_alias_edge_attributed(edge[0], edge[1])
			else:
				for edge in G.edges():
					alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			if is_attributed:
				for edge in G.edges():
					alias_edges[edge] = self.get_alias_edge_attributed(edge[0], edge[1])
					alias_edges[(edge[1], edge[0])] = self.get_alias_edge_attributed(edge[1], edge[0])
			else:
				for edge in G.edges():
					alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
					alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]