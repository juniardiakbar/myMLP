import random
import math


def split_data_set(data, fraction):
    train_set = data.sample(frac=0.8, random_state=0)
    test_set = data.drop(train_set.index)

    return train_set, test_set


def mat_vec(A, B):
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C


def vec_mat_bias(A, B, bias):
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C


def sigmoid(A, deriv=False):
    if deriv:
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A


def myMLP(df):
    target_attribute = list(df.columns)[-1]
    goal_idx = 0
    replace_goal = {}

    for val in df[target_attribute].unique():
        replace_goal[val] = goal_idx
        goal_idx += 1

    df[target_attribute].replace(replace_goal, inplace=True)
    data_train, data_test = split_data_set(df, .8)

    train_y = data_train[target_attribute]
    train_X = data_train.drop([target_attribute], axis=1)

    test_y = data_test[target_attribute]
    test_X = data_test.drop([target_attribute], axis=1)

    # Define parameter
    learning_rate = 0.05
    epoch = 600
    neuron = [4, 5, 3]

    weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
    weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
    bias = [0 for i in range(neuron[1])]
    bias_2 = [0 for i in range(neuron[2])]

    # Initiate weight with random between -1.0 ... 1.0
    for i in range(neuron[0]):
        for j in range(neuron[1]):
            weight[i][j] = 2 * random.random() - 1

    for i in range(neuron[1]):
        for j in range(neuron[2]):
            weight_2[i][j] = 2 * random.random() - 1

    train_X = train_X.values.tolist()
    train_y = train_y.values.tolist()

    e = 0
    while (e < epoch):

        cost_total = 0
        for (idx, inputs) in enumerate(train_X):
            net_1 = vec_mat_bias(inputs, weight, bias)
            out_1 = sigmoid(net_1)

            net_2 = vec_mat_bias(out_1, weight_2, bias_2)
            out_2 = sigmoid(net_2)

            target = [0, 0, 0]
            target[int(train_y[idx])] = 1

            # Cost function, Square Root Error
            error = 0
            for i in range(neuron[2]):
                error += (target[i] - out_2[i]) ** 2
            cost_total += error * 1 / neuron[2]

            # Backward propagation
            delta_2 = []
            for j in range(neuron[2]):
                delta_2.append(-1 * (target[j] - out_2[j]) *
                               out_2[j] * (1 - out_2[j]))  # * 2 / neuron[2]

            for i in range(neuron[1]):
                for j in range(neuron[2]):
                    weight_2[i][j] -= learning_rate * (delta_2[j] * out_1[i])
                    bias_2[j] -= learning_rate * delta_2[j]

            # Update weight and bias (layer 1)
            delta_1 = mat_vec(weight_2, delta_2)
            for j in range(neuron[1]):
                delta_1[j] = delta_1[j] * (out_1[j] * (1-out_1[j]))

            for i in range(neuron[0]):
                for j in range(neuron[1]):
                    weight[i][j] -= learning_rate * (delta_1[j] * inputs[i])
                    bias[j] -= learning_rate * delta_1[j]

        e += 1

    test_X = test_X.values.tolist()
    test_y = test_y.values.tolist()

    match = 0
    for (idx, inputs) in enumerate(test_X):
        net_1 = vec_mat_bias(inputs, weight, bias)
        out_1 = sigmoid(net_1)

        net_2 = vec_mat_bias(out_1, weight_2, bias_2)
        out_2 = sigmoid(net_2)

        print(out_2)

        print(out_2.index(max(out_2)), test_y[idx])
        if (out_2.index(max(out_2)) == test_y[idx]):
            match += 1

    print(match / len(test_X) * 100, "%")
