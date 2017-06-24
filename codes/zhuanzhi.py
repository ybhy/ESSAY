import numpy as np
weight = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print type(weight[0][0])
# print weight[0][0]
# print weight[0][:]
# print type(weight)
weight = np.array(weight)
# print type(weight)
# print len(weight)
for i in xrange(0, len(weight)):
	for j in xrange(0, len(weight)):
		if(i != j):
			xi = weight[i]
			xj = weight[j]
			# print (xi - xj)
			similarity = np.sqrt(np.dot(xi - xj, np.transpose(xi - xj)))
			print "%d %d %f " % (i, j, similarity)
