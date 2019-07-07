# RNN-recsys
实现一些基于RNN的推荐系统算法以及将rnn用于推荐系统的一些想法验证


### data
movielens-1m: 为了快速验证效果，采用movielens数据集
#### data split
train:  每个用户前N个月的数据
test：每个用户最后一个月的数据
#### train ba
* **train.txt**: each line is a training instance, in the form of user_history \t target_item_id \t label. user_history is a sequence of item_id, splited by space.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTMzMTk5ODMyLC0xNjU0NTE0ODA2XX0=
-->