import random
import math


def split_data_set(data, fraction):
    train_set = data.sample(frac=0.8, random_state=random.randrange(10))
    test_set = data.drop(train_set.index)

    return train_set, test_set


def calculate_net(node, weights, biases):
    nets = [0 for i in range(len(weights[0]))]

    for j in range(len(weights[0])):
        for k in range(len(weights)):
            nets[j] += node[k] * weights[k][j]

            if (len(biases) > 0):
                nets[j] += biases[j]

    return nets


def sigmoid(net):
    for i in range(len(net)):
        net[i] = 1 / (1 + math.exp(-net[i]))

    return net


def encode_target_attribute(df):
    target_attribute = list(df.columns)[-1]
    goal_idx = 0
    replace_goal = {}

    for val in df[target_attribute].unique():
        replace_goal[val] = goal_idx
        goal_idx += 1

    df[target_attribute].replace(replace_goal, inplace=True)
    return df, goal_idx


def split_dataset_to_train_and_test(df):
    data_train, data_test = split_data_set(df, .8)

    target_attribute = list(df.columns)[-1]

    train_y = data_train[target_attribute]
    train_X = data_train.drop([target_attribute], axis=1)

    test_y = data_test[target_attribute]
    test_X = data_test.drop([target_attribute], axis=1)
    return train_X, train_y, test_X, test_y


def split_trainset_into_chunks(train_X, train_y, batch_size):
    train_X = train_X.values.tolist()
    train_y = train_y.values.tolist()

    train_X_chunk_list = [train_X[batch_size*i:min(batch_size*i+batch_size, len(train_X))]
                          for i in range(len(train_X)//batch_size)]
    train_Y_chunk_list = [train_y[batch_size*i:min(batch_size*i+batch_size, len(train_y))]
                          for i in range(len(train_y)//batch_size)]

    return train_X_chunk_list, train_Y_chunk_list


def initialize_weight_and_biases(neuron):
    weights = []
    biases = []

    for i in range(len(neuron) - 1):
        w = [[0 for k in range(neuron[i + 1])] for j in range(neuron[i])]
        b = [0 for j in range(neuron[i + 1])]

        weights.append(w)
        biases.append(b)

    # Initiate weight with random between -1.0 ... 1.0
    for i in range(len(neuron) - 1):
        for j in range(neuron[i]):
            for k in range(neuron[i + 1]):
                weights[i][j][k] = 2 * random.random() - 1
    return weights, biases


def predict(test_X, test_y, neuron, weights, biases):
    test_X = test_X.values.tolist()
    test_y = test_y.values.tolist()

    match = 0
    for (idx, inputs) in enumerate(test_X):
        nets = []
        outs = []

        for i in range(len(neuron) - 1):
            if (i == 0):
                n = vec_mat_bias(inputs, weights[i], biases[i])
            else:
                n = vec_mat_bias(outs[i - 1], weights[i], biases[i])

            nets.append(n)
            o = sigmoid(n)
            outs.append(o)

        if (outs[-1].index(max(outs[-1])) == test_y[idx]):
            match += 1

    return match/len(test_X)


def myMLP(df, learning_rate=0.05, max_iteration=600, hidden_layer=[3], batch_size=10):
    df, goal_idx = encode_target_attribute(df)

    # Splitting dataset into training and testing
    train_X, train_y, test_X, test_y = split_dataset_to_train_and_test(df)

    # Define parameter
    neuron = [len(train_X.columns)] + hidden_layer + [goal_idx]

    # Splitting into list of mini-batch
    train_X_chunk_list, train_Y_chunk_list = split_trainset_into_chunks(
        train_X, train_y, batch_size)

    # Initialize weights and biases
    weights, biases = initialize_weight_and_biases(neuron)

    for i in range(max_iteration):
        cost_total = 0
        for idx_batch in range(len(train_X_chunk_list)):

            for (idx, inputs) in enumerate(train_X_chunk_list[idx_batch]):
                nets = []
                outs = []

                for i in range(len(neuron) - 1):
                    if (i == 0):
                        n = calculate_net(inputs, weights[i], biases[i])
                    else:
                        n = calculate_net(outs[i - 1], weights[i], biases[i])

                    nets.append(n)
                    o = sigmoid(n)
                    outs.append(o)

                target = [0 for i in range(neuron[-1])]
                target[int(train_Y_chunk_list[idx_batch][idx])] = 1

                # Cost function, Square Root Error
                error = 0
                for i in range(neuron[-1]):
                    error += (target[i] - outs[-1][i]) ** 2
                cost_total += error * 1 / neuron[2]

                # Backward propagation
                deltas = [[] for i in range(len(neuron) - 1)]

                i = len(neuron) - 2
                while (i >= 0):

                    d = []
                    if (i == len(neuron) - 2):
                        for j in range(neuron[i + 1]):
                            d.append(-1 * (target[j] - outs[i][j]) *
                                     outs[i][j] * (1 - outs[i][j])) * 2. / neuron[-1]
                    else:
                        out = calculate_net(weights[i + 1], deltas[i + 1], [])

                        for j in range(neuron[i + 1]):
                            d.append(out[j] * outs[i][j] * (1 - outs[i][j]))

                    deltas[i] = d

                    for j in range(neuron[i]):
                        for k in range(neuron[i+1]):
                            if (i == 0):
                                weights[i][j][k] -= learning_rate * \
                                    (deltas[i][k] * inputs[j])
                            else:
                                weights[i][j][k] -= learning_rate * \
                                    (deltas[i][k] * outs[i - 1][j])

                            biases[i][k] -= learning_rate * deltas[i][k]

                    i -= 1

    print(predict(test_X, test_y, neuron, weights, biases) * 100, "%")
