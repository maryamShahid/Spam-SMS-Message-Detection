import pandas
import numpy
import math

sms_train_features_path = 'datasets/sms_train_features.csv'
sms_train_features = pandas.read_csv(sms_train_features_path, index_col=0, skiprows=0)

sms_train_labels_path = 'datasets/sms_train_labels.csv'
sms_train_labels = pandas.read_csv(sms_train_labels_path, index_col=0, skiprows=0)

sms_test_features_path = 'datasets/sms_test_features.csv'
sms_test_features = pandas.read_csv(sms_test_features_path, index_col=0, skiprows=0)

sms_test_labels_path = 'datasets/sms_test_labels.csv'
sms_test_labels = pandas.read_csv(sms_test_labels_path, index_col=0, skiprows=0)

vocab_path = 'datasets/vocabulary.txt'
vocabulary = numpy.loadtxt(vocab_path, dtype=str)

train_features_matrix = sms_train_features.to_numpy()
train_labels_matrix = sms_train_labels.to_numpy()
test_features_matrix = sms_test_features.to_numpy()
test_labels_matrix = sms_test_labels.to_numpy()

spamCount = sms_train_labels.value_counts().to_numpy()[1]
normalCount = sms_train_labels.value_counts().to_numpy()[0]
vocabCount = 3458


def calculatePredictions(x, y, labels):
    prediction_matrix = (x > y)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    labels = labels.flatten()
    for i in range(len(prediction_matrix)):
        if prediction_matrix[i] == 1 and labels[i] == 1:
            true_positive += 1
        elif prediction_matrix[i] == 0 and labels[i] == 0:
            true_negative += 1
        elif prediction_matrix[i] == 1 and labels[i] == 0:
            false_positive += 1
        elif prediction_matrix[i] == 0 and labels[i] == 1:
            false_negative += 1
    print(numpy.array([[true_positive, false_negative], [false_positive, true_negative]]))
    print((true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive) * 100,
          'accuracy')


def calculateLoggedPrior(x, y):
    return math.log(x / (x + y))


def calculateXgivenY(x, y):
    total_count_of_x_in_y = (numpy.matmul(x, y)) + alpha
    sum_of_total_count_of_x_in_y = numpy.sum(total_count_of_x_in_y)
    total_count_of_x_and_y = sum_of_total_count_of_x_in_y + alpha * vocabulary.shape[0]
    return total_count_of_x_in_y / total_count_of_x_and_y


def calculateXgivenYBernoulli(x, y):
    total_count_of_x_in_y = (numpy.matmul(x.T, y))
    sum_of_total_count_of_x_in_y = numpy.sum(total_count_of_x_in_y)
    return total_count_of_x_in_y / sum_of_total_count_of_x_in_y


def calculateYgivenXBernoulli(x, y):
    P_X_given_Y = (numpy.multiply(x.T, y))
    one_Minus_P_X_given_Y = numpy.multiply(1 - x.T, 1 - y)
    return P_X_given_Y + one_Minus_P_X_given_Y


logPriorSpam = calculateLoggedPrior(spamCount, normalCount)
logPriorNormal = calculateLoggedPrior(normalCount, spamCount)

#  Multinomial Naive Bayes Model

# Classification
alpha = 1
# P ( X | Y = spam)
X_given_Y_is_spam = calculateXgivenY(train_features_matrix.T, train_labels_matrix)
# P ( X | Y = normal)
X_given_Y_is_normal = calculateXgivenY(train_features_matrix.T, (1 - train_labels_matrix))

logged_X_given_Y_is_spam = numpy.log(X_given_Y_is_spam)
logged_X_given_Y_is_normal = numpy.log(X_given_Y_is_normal)

# P ( Y = spam | X ) = P ( X | Y = spam) * P ( Y = spam)
Y_given_X_spam = numpy.multiply(test_features_matrix, logged_X_given_Y_is_spam.T)
spam = logPriorSpam + numpy.sum(Y_given_X_spam, axis=1)

# P ( Y = normal | X ) = P ( X | Y = normal) * P ( Y = normal)
Y_given_X_normal = numpy.multiply(test_features_matrix, logged_X_given_Y_is_normal.T)
normal = logPriorNormal + numpy.sum(Y_given_X_normal, axis=1)

calculatePredictions(spam, normal, test_labels_matrix)

#   Bernoulli Naive Bayes Model

# convert train features to 1 and 0
train_features_matrix_bernoulli = train_features_matrix.copy()
train_features_matrix_bernoulli[train_features_matrix_bernoulli != 0] = 1

# Classification
# P ( X | Y = spam)
theta_spam = calculateXgivenYBernoulli(train_features_matrix_bernoulli, train_labels_matrix)
# P ( X | Y = normal)
theta_normal = calculateXgivenYBernoulli(train_features_matrix_bernoulli, 1 - train_labels_matrix)

# P ( Y = spam | X ) = P ( X | Y = spam) * P ( Y = spam)
totalSpamProb = numpy.log(calculateYgivenXBernoulli(theta_spam, test_features_matrix))
spam = numpy.nansum(totalSpamProb, axis=1)

# P ( Y = normal | X ) = P ( X | Y = normal) * P ( Y = normal)
totalNormalProb = numpy.log(calculateYgivenXBernoulli(theta_normal, test_features_matrix))
normal = numpy.nansum(totalNormalProb, axis=1)

calculatePredictions(spam, normal, test_labels_matrix)

# Mutual Information feature selection
highestProbabilities = numpy.sum(train_features_matrix_bernoulli, axis=0)

indices = numpy.argpartition(highestProbabilities, -100)[-100:]
indices = indices[numpy.argsort(highestProbabilities[indices])]
print(indices)

for i in range(indices.size):
    feature = train_features_matrix_bernoulli[:, indices[i]]

for i in range(indices.size):
    labels = train_labels_matrix.T[:, indices[i]]

# Classification
# P ( X | Y = spam)
theta_spam = calculateXgivenYBernoulli(train_features_matrix_bernoulli, train_labels_matrix)
# P ( X | Y = normal)
theta_normal = calculateXgivenYBernoulli(train_features_matrix_bernoulli, 1 - train_labels_matrix)

# P ( Y = spam | X ) = P ( X | Y = spam) * P ( Y = spam)
totalSpamProb = numpy.log(calculateYgivenXBernoulli(theta_spam, test_features_matrix))
spam = numpy.nansum(totalSpamProb, axis=1)

# P ( Y = normal | X ) = P ( X | Y = normal) * P ( Y = normal)
totalNormalProb = numpy.log(calculateYgivenXBernoulli(theta_normal, test_features_matrix))
normal = numpy.nansum(totalNormalProb, axis=1)

calculatePredictions(spam, normal, test_labels_matrix)

