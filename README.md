# RNN-recsys
实现一些基于RNN的推荐系统算法以及将rnn用于推荐系统的一些想法验证


### data
movielens-1m: 为了快速验证效果，采用movielens数据集
#### data split
train:  每个用户前N个月的数据
test：每个用户最后一个月的数据
#### train batches
input : 上一个月的历史数据
output: 下一个月的电影数据

<!--stackedit_data:
eyJoaXN0b3J5IjpbNDQwMzIzMjk0LC0xNjU0NTE0ODA2XX0=
-->