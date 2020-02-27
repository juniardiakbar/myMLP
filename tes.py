import random
import math


def split_data_set(data, fraction):
    train_set = data.sample(frac=0.8, random_state=random.randrange(10))
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


def myMLP(df, learning_rate=0.05, max_iteration=600, hidden_layer=[3]):
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

    # # Define parameter
    # neuron = hidden_layer
    # neuron.insert(0, len(train_X.columns))
    # neuron.append(goal_idx)

    neuron = [2, 2, 2]

    train_X = train_X.values.tolist()
    train_y = train_y.values.tolist()

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

    weights = [[[.15, .25, ], [.2, .3]], [[.4, .5, ], [.45, .55]]]
    biases = [[.175, .175], [.3, .3]]

    e = 0
    while (e < max_iteration):

        cost_total = 0
        for (idx, inputs) in enumerate(train_X):
            nets = []
            outs = []

            inputs = [0.05, 0.1]

            for i in range(len(neuron) - 1):
                if (i == 0):
                    n = vec_mat_bias(inputs, weights[i], biases[i])
                else:
                    n = vec_mat_bias(outs[i - 1], weights[i], biases[i])

                o = sigmoid(n)

                nets.append(n)
                outs.append(o)

            # target = [0 for i in range(neuron[-1])]
            # target[int(train_y[idx])] = 1

            target = [.01, .99]

            # Cost function, Square Root Error
            error = 0
            for i in range(neuron[-1]):
                error += (target[i] - outs[-1][i]) ** 2

            cost_total += error * 1 / neuron[2]

            print(cost_total)

            # Backward propagation
            deltas = [[] for i in range(len(neuron) - 1)]

            i = len(neuron) - 2
            while (i >= 0):

                if (i == len(neuron) - 2):
                    d = []

                    for j in range(neuron[i + 1]):
                        d.append(-1 * (target[j] - outs[i][j]) *
                                 outs[i][j] * (1 - outs[i][j]))  # * 2 / neuron[2]
                else:
                    d = mat_vec(weights[i + 1], deltas[i + 1])

                    for j in range(neuron[i + 1]):
                        d[j] = d[j] * (outs[i][j] * (1 - outs[i][j]))

                deltas[i] = d

                for j in range(neuron[i]):
                    for k in range(neuron[i+1]):
                        if (i == 0):
                            weights[i][j][k] -= learning_rate * \
                                (deltas[i][k] * inputs[j])
                        else:
                            weights[i][j][k] -= learning_rate * \
                                (deltas[i][k] * outs[i - 1][j])

                        # biases[i][k] -= learning_rate * deltas[i][k]

                i -= 1

            break

        e += 1

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

    print(match / len(test_X) * 100, "%")
