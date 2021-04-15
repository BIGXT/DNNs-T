import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
# 读取序列数据
data = pd.read_excel("seq_data.xlsx")
# 取前800个
data = data.values[1:800]
# 标准化数据
normalize_data = (data - np.mean(data)) / np.std(data)
# normalize_data = normalize_data[:, np.newaxis]
# data=data[:, np.newaxis]
s = np.std(data)
m = np.mean(data)
# 序列段长度
time_step = 96
# 隐藏层节点数目
rnn_unit = 8
# cell层数
lstm_layers = 2
# 序列段批处理数目
batch_size = 7
# 输入维度
input_size = 1
# 输出维度
output_size = 1
# 学习率
lr = 0.006
train_x, train_y = [], []
for i in range(len(data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

X = tf.placeholder(tf.float32, [None, time_step, input_size])  # shape(?,time_step, input_size)
Y = tf.placeholder(tf.float32, [None, time_step, output_size])  # shape(?,time_step, out_size)

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


def lstm(batch):
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(rnn_unit) for i in range(lstm_layers)])
    init_state = cell.zero_state(batch, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    loss_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):  # We can increase the number of iterations to gain better result.
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = end + batch_size
            loss_list.append(loss_)
            if i % 10 == 0:
                print("Number of iterations:", i, " loss:", loss_list[-1])
                if i > 0 and loss_list[-2] > loss_list[-1]:
                    saver.save(sess, 'model_save1\\modle.ckpt')
        # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        # if you run it in Linux,please use  'model_save1/modle.ckpt'
        print("The train has finished")


train_lstm()


def prediction():
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(1)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, 'model_save1\\modle.ckpt')
        # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        # if you run it in Linux,please use  'model_save1/modle.ckpt'
        predict = []
        for i in range(0, np.shape(train_x)[0]):
            next_seq = sess.run(pred, feed_dict={X: [train_x[i]]})
            predict.append(next_seq[-1])
        plt.figure()
        plt.plot(list(range(len(data))), data, color='b')
        plt.plot(list(range(time_step + 1, np.shape(train_x)[0] + 1 + time_step)), [value * s + m for value in predict],
                 color='r')
        plt.show()


prediction()
'''

def l1(x_train, x_val , y_train, y_val):
    # 导入 keras 相关模块
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    import numpy as np
    # 指定数据维度和时间步等参数
    data_dim = 2
    timesteps = 2
    num_classes = 2
    # 搭建一个 LSTM 多分类过程
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    """
    # 生成模拟训练数据
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, num_classes))
    # 生成模型验证数据
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, num_classes))
    """

    # 转换数据集
    x_train = chunks(np.array(x_train), timesteps)
    x_val = chunks(np.array(x_val), timesteps)

    y_train = list(map(str, y_train))
    y_val = list(map(str, y_val))
    y_train = chunks(np.array(y_train), timesteps)
    y_val = chunks(np.array(y_val), timesteps)

    # 模型训练
    model.fit(x_train, y_train, batch_size=64, epochs=500, validation_data=(x_val, y_val))
    model.summary()

def chunks(l, n):
    arr= [l[i:i+n] for i in range(0, len(l), n)]
    return np.array(arr[:-1])


def l2():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    # 定义常量
    rnn_unit = 10  # hidden layer units
    input_size = 2
    output_size = 1
    lr = 0.0006  # 学习率
    # ——————————————————导入数据——————————————————————
    f = open('X1.csv')
    df = pd.read_csv(f)  # 读入股票数据
    data = df.iloc[:, 2:10].values  # 取第3-10列

    # 获取训练集
    def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=5800):
        batch_index = []
        data_train = data[train_begin:train_end]
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
        train_x, train_y = [], []  # 训练集
        for i in range(len(normalized_train_data) - time_step):
            if i % batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + time_step, :7]
            y = normalized_train_data[i:i + time_step, 7, np.newaxis]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data) - time_step))
        return batch_index, train_x, train_y

    # 获取测试集
    def get_test_data(time_step=20, test_begin=5800):
        data_test = data[test_begin:]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test - mean) / std  # 标准化
        size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
        test_x, test_y = [], []
        for i in range(size - 1):
            x = normalized_test_data[i * time_step:(i + 1) * time_step, :7]
            y = normalized_test_data[i * time_step:(i + 1) * time_step, 7]
            test_x.append(x.tolist())
            test_y.extend(y)
        test_x.append((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
        test_y.extend((normalized_test_data[(i + 1) * time_step:, 7]).tolist())
        return mean, std, test_x, test_y

    # ——————————————————定义神经网络变量——————————————————
    # 输入层、输出层权重、偏置

    weights = {
        'in': tf.Variable(tf.random.normal([input_size, rnn_unit])),
        'out': tf.Variable(tf.random.normal([rnn_unit, 1]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

    # ——————————————————定义神经网络变量——————————————————
    def lstm(X):
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = weights['in']
        b_in = biases['in']
        input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                     dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

    # ——————————————————训练模型——————————————————
    def train_lstm(batch_size=80, time_step=15, train_begin=2000, train_end=5800):
        X = tf.compat.v1.placeholder(tf.float32, shape=[None, time_step, input_size])
        Y = tf.compat.v1.placeholder(tf.float32, shape=[None, time_step, output_size])
        batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
        pred, _ = lstm(X)
        # 损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        module_file = tf.train.latest_checkpoint()
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess, module_file)
            # 重复训练10000次
            for i in range(2000):
                for step in range(len(batch_index) - 1):
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                   Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print(i, loss_)
                if i % 200 == 0:
                    print("保存模型：", saver.save(sess, 'stock2.model', global_step=i))

    train_lstm()

    # ————————————————预测模型————————————————————
    def prediction(time_step=20):
        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        # Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
        mean, std, test_x, test_y = get_test_data(time_step)
        pred, _ = lstm(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint()
            saver.restore(sess, module_file)
            test_predict = []
            for step in range(len(test_x) - 1):
                prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)
            test_y = np.array(test_y) * std[7] + mean[7]
            test_predict = np.array(test_predict) * std[7] + mean[7]
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
            # 以折线图表示结果
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b')
            plt.plot(list(range(len(test_y))), test_y, color='r')
            plt.show()

    prediction()