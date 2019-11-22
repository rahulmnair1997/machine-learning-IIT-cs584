import numpy
import pandas
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm

data = pd.read_csv("/Users/rahulnair/desktop/SpiralWithCluster.csv")
cluster_1 = data.groupby('SpectralCluster').size()/data.shape[0]
print("Percent of the observations have SpectralCluster", cluster_1)
xVar = pd.DataFrame(data, columns = ['x', 'y'])
y = data['SpectralCluster']

# for the 'relu' function


def Build_NN_relu (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'relu', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    thisFit = nnObj.fit(xVar, y)
    y_predProb = nnObj.predict_proba(xVar)
    y_pred = numpy.where(y_predProb[:,1] >= 0.5, 1, 0)

    Loss = nnObj.loss_
    acc = 1 - metrics.accuracy_score(y, y_pred)
    activation = 'relu'
    niter = nnObj.n_iter_
    return (Loss, acc, activation, niter)

result_relu = pd.DataFrame(columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"])
for i in numpy.arange(1,6):
    for j in numpy.arange(1,11,1):
        Loss, acc, activation, niter = Build_NN_relu (nLayer = i, nHiddenNeuron = j)
        result_relu = result_relu.append(pandas.DataFrame([[i, j, Loss, acc, activation, niter]],
                               columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"]),sort = False)


record_relu = result_relu[result_relu['Loss'] == min(result_relu['Loss'])]

# for 'identity' activation function


def Build_NN_identity (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'identity', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    thisFit = nnObj.fit(xVar, y)
    y_predProb = nnObj.predict_proba(xVar)
    y_pred = numpy.where(y_predProb[:,1] >= 0.5, 1, 0)

    Loss = nnObj.loss_
    acc = 1 - metrics.accuracy_score(y, y_pred)
    activation = 'identity'
    niter = nnObj.n_iter_
    return (Loss, acc, activation, niter)


result_identity = pd.DataFrame(columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"])
for i in numpy.arange(1,6):
    for j in numpy.arange(1,11,1):
        Loss, acc, activation, niter = Build_NN_identity (nLayer = i, nHiddenNeuron = j)
        result_identity = result_identity.append(pandas.DataFrame([[i, j, Loss, acc, activation, niter]],
                               columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"]),sort = False)

record_identity = result_identity[result_identity['Loss'] == min(result_identity['Loss'])]

# for 'tanh' activation function


def Build_NN_tanh (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'tanh', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    thisFit = nnObj.fit(xVar, y)
    y_predProb = nnObj.predict_proba(xVar)
    y_pred = numpy.where(y_predProb[:,1] >= 0.5, 1, 0)
    Loss = nnObj.loss_
    acc = 1 - metrics.accuracy_score(y, y_pred)
    activation = 'tanh'
    niter = nnObj.n_iter_
    return (Loss, acc, activation, niter)


result_tanh = pd.DataFrame(columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"])
for i in numpy.arange(1,6):
    for j in numpy.arange(1,11,1):
        Loss, acc, activation, niter = Build_NN_tanh (nLayer = i, nHiddenNeuron = j)
        result_tanh = result_identity.append(pandas.DataFrame([[i, j, Loss, acc, activation, niter]],
                               columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"]),sort = False)

record_tanh = result_tanh[result_tanh['Loss'] == min(result_tanh['Loss'])]

# for 'logistic' activation function


def Build_NN_log (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'logistic', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    thisFit = nnObj.fit(xVar, y)
    y_predProb = nnObj.predict_proba(xVar)
    y_pred = numpy.where(y_predProb[:,1] >= 0.50, 1, 0)
    Loss = nnObj.loss_
    acc = 1 - metrics.accuracy_score(y, y_pred)
    activation = 'logistic'
    return (Loss, acc, activation, niter)


result_log = pd.DataFrame(columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"])
for i in numpy.arange(1,6):
    for j in numpy.arange(1,11,1):
        Loss, acc, activation, niter = Build_NN_log (nLayer = i, nHiddenNeuron = j)
        result_log = result_log.append(pandas.DataFrame([[i, j, Loss, acc, activation, niter]],
                               columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Misclassification', 'Function', "Number of iterations"]),sort = False)

record_log = result_log[result_log['Loss'] == min(result_log['Loss'])]
table = [record_identity, record_log, record_relu, record_tanh]
result = pd.concat(table)
print(result)

# Build Neural Network
nnObj = nn.MLPClassifier(hidden_layer_sizes = (8,)* 4,
                        activation = 'relu', verbose = False,
                        solver = 'lbfgs', learning_rate_init = 0.1,
                        max_iter = 5000, random_state = 20191108)
thisFit = nnObj.fit(xVar, y)
y_predProb_1 = nnObj.predict_proba(xVar)
data['pred'] = numpy.where(y_predProb_1[:,1] >= 0.5, 1, 0)
out_fun = nnObj.out_activation_
print("the activation function of output layer", out_fun)
result_final = result[result['Loss'] == min(result['Loss'])]
print(result_final)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = data[data['pred'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('MLP (4 Layers, 8 Neurons)')
    plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
data['y_predProb'] = y_predProb_1[:,1]
qwe = data.groupby('SpectralCluster')
print("Mean: ", qwe.get_group(1).describe()['y_predProb']['mean'])
print("Standard Deviation", qwe.get_group(1).describe()['y_predProb']['std'])
print("count: ", qwe.get_group(1).describe()['y_predProb']['count'])
svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(xVar, y)
y_predictClass = thisFit.predict(xVar)
print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)
mis = 1 - metrics.accuracy_score(y, y_predictClass)
print("Misclassification rate: ", mis)

# get the separating hyperplane
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = numpy.linspace(-5, 5)
yy = a * xx - (thisFit.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = thisFit.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = thisFit.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

data['_PredictedClass_SVM'] = y_predictClass

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = data[data['_PredictedClass_SVM'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.plot(xx, yy, color = 'black', linestyle = '--')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Convert to the polar coordinates
data['radius'] = numpy.sqrt(data['x']**2 + data['y']**2)
data['theta'] = numpy.arctan2(data['y'], data['x'])


def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)


data['theta'] = data['theta'].apply(customArcTan)

# Build Support Vector Machine classifier
xTrain = data[['radius','theta']]
yTrain = data['SpectralCluster']
carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = data[data['SpectralCluster'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('Radius')
plt.ylabel('Angle in Radians')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)
data_1 = data.loc[(data['radius']<= 1.5) & (data['theta']>=6) & (data['SpectralCluster'] == 0)]
data_2 = data.loc[(data['radius']<= 3) & (data['theta']>=3) & (data['SpectralCluster'] == 1)]
data_3 = data.loc[(data['radius']<= 4) & (data['theta']<=6.2) & (data['SpectralCluster'] == 0)]
data_4 = data.loc[(data['radius']<= 5) & (data['theta']<=3.1 ) & (data['SpectralCluster'] == 1)]
data_1['Group'] = 0
data_2['Group'] = 1
data_3['Group'] = 2
data_4['Group'] = 3
qwerty = pd.concat([data_1, data_2, data_3, data_4])
carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = qwerty[qwerty['Group'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('Radius')
plt.ylabel('Angle in Radians')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
gr0_gr1 = qwerty.loc[(qwerty['Group'] == 0) | (qwerty['Group'] == 1)]
gr1_gr2 = qwerty.loc[(qwerty['Group'] == 1) | (qwerty['Group'] == 2)]
gr2_gr3 = qwerty.loc[(qwerty['Group'] == 2) | (qwerty['Group'] == 3)]
x_train01 = gr0_gr1[['radius', 'theta']]
yTrain01 = gr0_gr1['Group']
svm_Model01 = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit01 = svm_Model01.fit(x_train01, yTrain01)
y_predictClass01 = thisFit01.predict(x_train01)
print('Intercept for group 0 and 1= ', thisFit01.intercept_)
print('Coefficients for group 0 and 1= ', thisFit01.coef_)

# get the separating hyperplane
w01 = thisFit01.coef_[0]
a01 = -w01[0] / w01[1]
xx01 = numpy.linspace(1, 3)
yy01 = a01 * xx01 - (thisFit01.intercept_[0]) / w01[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b01 = thisFit01.support_vectors_[0]
yy_down01 = a01 * xx01 + (b01[1] - a01 * b01[0])

b01 = thisFit01.support_vectors_[-1]
yy_up01 = a01 * xx01 + (b01[1] - a01 * b01[0])

x_train12 = gr1_gr2[['radius', 'theta']]
yTrain12 = gr1_gr2['Group']
svm_Model12 = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit12 = svm_Model12.fit(x_train12, yTrain12)
y_predictClass12 = thisFit12.predict(x_train12)
print('Intercept for group 1 and 2= ', thisFit12.intercept_)
print('Coefficients for group 1 and 2 = ', thisFit12.coef_)

# get the separating hyperplane
w12 = thisFit12.coef_[0]
a12 = -w12[0] / w12[1]
xx12 = numpy.linspace(1, 4)
yy12 = a12 * xx12 - (thisFit12.intercept_[0]) / w12[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b12 = thisFit12.support_vectors_[0]
yy_down12 = a12 * xx12 + (b12[1] - a12 * b12[0])

b12 = thisFit12.support_vectors_[-1]
yy_up12 = a12 * xx12 + (b12[1] - a12 * b12[0])

x_train23 = gr2_gr3[['radius', 'theta']]
yTrain23 = gr2_gr3['Group']
svm_Model23 = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit23 = svm_Model23.fit(x_train23, yTrain23)
y_predictClass23 = thisFit23.predict(x_train23)
print('Intercept for group 2 and 3 = ', thisFit23.intercept_)
print('Coefficients for group 2 and 3 = ', thisFit23.coef_)

# get the separating hyperplane
w23 = thisFit23.coef_[0]
a23 = -w23[0] / w23[1]
xx23 = numpy.linspace(1, 5)
yy23 = a23 * xx23 - (thisFit23.intercept_[0]) / w23[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b23 = thisFit23.support_vectors_[0]
yy_down23 = a23 * xx23 + (b23[1] - a23 * b23[0])

b23 = thisFit23.support_vectors_[-1]
yy_up23 = a23 * xx23 + (b23[1] - a23 * b23[0])

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = qwerty[qwerty['Group'] == i]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = i, s = 25)
plt.plot(xx01, yy01, color = 'black', linestyle = '--')
plt.plot(xx12, yy12, color = 'black', linestyle = '--')
plt.plot(xx23, yy23, color = 'black', linestyle = '--')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.xlim([1, 5])
plt.ylim([-1, 7])
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

h0_xx = xx01 * numpy.cos(yy01)
h0_yy = xx01 * numpy.sin(yy01)

h1_xx = xx12 * numpy.cos(yy12)
h1_yy = xx12 * numpy.sin(yy12)

h2_xx = xx23 * numpy.cos(yy23)
h2_yy = xx23 * numpy.sin(yy23)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = data[data['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.plot(h0_xx , h0_yy, color = 'black', linestyle = '--')
plt.plot(h1_xx, h1_yy, color = 'black', linestyle = '--')
plt.plot(h2_xx, h2_yy, color = 'black', linestyle = '--')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# hyperplane for group 0 and 1 has been removed since it is touching the clusters
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = data[data['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.plot(h1_xx, h1_yy, color = 'black', linestyle = '--')
plt.plot(h2_xx, h2_yy, color = 'black', linestyle = '--')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

