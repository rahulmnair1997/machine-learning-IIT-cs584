
# AUTHOR- RAHUL NAIR (A20438470)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier

# load the data
qwe = pd.read_csv('/Users/rahulnair/desktop/NormalSample.csv')
N = qwe['x'].count()

# calculate N
print('Total number of samples',N)

# calculating IQR
iqr = np.subtract(*np.percentile(qwe['x'], [75,25]))

# For Question 1
print("      FOR QUESTION 1     ")
# Izenman Function
h = (2 * iqr)/(N)**(1/3)
print('Recommended h value',h)

# print minimum and maximum value
minimum = qwe['x'].min()
maximum = qwe['x'].max()
print('Minimum and maximum value is',minimum, maximum)

# calculating density estimator


def density_estimator(midpoints, h):
    estimator = []
    qwerty = sorted(qwe['x'])
    for j in midpoints:
        counter = 0
        for a in qwerty:
            if a>=(j - (h/2)):
                if a<=(j + (h/2)):
                    counter+=1
                else:
                    break
        estimator.append(counter/(N*h))
    for (l, r) in zip(midpoints, estimator):
        print(l, r)

# For calculating mid-points


def mid_point(binw):
    a = 26
    b = 36
    bins = np.arange(a, b + (binw/2), binw)
    y, sides = np.histogram(qwe['x'], bins)
    midpoints = 0.5*(sides[1:] + sides[:-1])
    print("DENSITY ESTIMATOR FOR ", binw)
    density_estimator(midpoints,binw)


# For binw = 0.1
binw = 0.1
mid_point(binw)
a = 26
b = 36
plt.hist(qwe['x'], bins = np.arange(a, b+(binw/2), binw))
plt.xlabel("Mid-point")
plt.ylabel("Estimator")
plt.show()

# for binw = 0.5
binw2 = 0.5
mid_point(binw2)
plt.hist(qwe['x'], bins = np.arange(a, b + (binw2/2), binw2))
plt.xlabel("Mid-point")
plt.ylabel("Estimator")
plt.show()

# for binw = 1
binw3 = 1
mid_point(binw3)
plt.hist(qwe['x'], bins = np.arange(a, b+(binw3/2), binw3))
plt.xlabel("Mid-point")
plt.ylabel("Estimator")
plt.show()

# for binw = 2
binw4 = 2
mid_point(binw4)
plt.hist(qwe['x'], bins = np.arange(a, b+(binw4/2), binw4))
plt.xlabel("Mid-point")
plt.ylabel("Estimator")
plt.show()


# For Question 2
print("      FOR QUESTION 2     ")
print("for Main Data")
print(qwe['x'].describe())

# Dividing the data into 2 groups
df = pd.DataFrame(qwe)
df0 = df[df.group==0]
df1 = df[df.group==1]
print("for  Data of group 0")
print(df0['x'].describe())
print("for  Data of group 1")
print(df1['x'].describe())

# Calculate iqr
iqr = np.subtract(*np.percentile(qwe['x'],[75,25]))
print('IQR = ',iqr)
plt.boxplot(qwe['x'])
plt.show()
fig, ax = plt.subplots()
bp1 = ax.boxplot(qwe['x'], positions = [1], notch = True, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C0"))
bp2 = ax.boxplot(df0['x'], positions = [2], notch = True, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C2"))
bp3 = ax.boxplot(df1['x'], positions = [3], notch = True, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C4"))
ax.set_xlim(0,6)
plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['Ungrouped', 'Grouped by 0', 'Grouped by 1'],loc='upper right')
plt.show()


# For Question 3
print("      FOR QUESTION 3     ")
data = pd.read_csv('/Users/rahulnair/desktop/Fraud.csv')
df = pd.DataFrame(data)
df1 = df[df.FRAUD==1]
fraud_cases = df1['FRAUD'].count()
Total_cases = data['FRAUD'].count()

# print percentage of fraud cases
print('Percentage of fraud cases found ',(fraud_cases/Total_cases) * 100)
df2 = df[df.FRAUD==0]

fig, ax = plt.subplots()
bp1 = ax.boxplot(df2['TOTAL_SPEND'], positions = [1], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C0"))
bp2 = ax.boxplot(df1['TOTAL_SPEND'], positions = [2], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C2"))
ax.set_ylim(0,6)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Not Fraud', 'Fraud'],loc='upper right')
plt.show()

fig, ax = plt.subplots()
bp1 = ax.boxplot(df2['DOCTOR_VISITS'], positions = [1], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C0"))
bp2 = ax.boxplot(df1['DOCTOR_VISITS'], positions = [2], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C2"))
ax.set_ylim(0,6)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Not Fraud', 'Fraud'],loc='upper right')
plt.show()

fig, ax = plt.subplots()
bp1 = ax.boxplot(df2['NUM_CLAIMS'], positions = [1], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C0"))
bp2 = ax.boxplot(df1['NUM_CLAIMS'], positions = [2], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C2"))
ax.set_ylim(0,6)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Not Fraud', 'Fraud'],loc='upper right')
plt.show()

fig, ax = plt.subplots()
bp1 = ax.boxplot(df2['MEMBER_DURATION'], positions = [1], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C0"))
bp2 = ax.boxplot(df1['MEMBER_DURATION'], positions = [2], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C2"))
ax.set_ylim(0,6)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Not Fraud', 'Fraud'],loc='upper right')
plt.show()

fig, ax = plt.subplots()
bp1 = ax.boxplot(df2['OPTOM_PRESC'], positions = [1], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C0"))
bp2 = ax.boxplot(df1['OPTOM_PRESC'], positions = [2], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C2"))
ax.set_ylim(0,6)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Not Fraud', 'Fraud'],loc='upper right')
plt.show()

fig, ax = plt.subplots()
bp1 = ax.boxplot(df2['NUM_MEMBERS'], positions = [1], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C0"))
bp2 = ax.boxplot(df1['NUM_MEMBERS'], positions = [2], vert = False, widths = 0.40, patch_artist = True, boxprops = dict(facecolor="C2"))
ax.set_ylim(0,6)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Not Fraud', 'Fraud'],loc='upper right')
plt.show()

x = np.genfromtxt('/Users/rahulnair/desktop/Fraud.csv', delimiter=',', skip_header=1, usecols = (2,3,4,5,6,7))
xtx = np.mat(x.transpose()) * np.mat(x)

# Calculating eigenvectors and eigenvalues
evals, evacs = LA.eigh(xtx)
print('Eigenvalues of x=\n', evals)
print('Eigenvectors of x=\n', evacs)
transf = evacs * LA.inv(np.sqrt(np.diagflat(evals)))
print("Transformation matrix=\n",transf)
transf_x = x * transf
xtx = transf_x.transpose() * transf_x
print("Proof that the transformed matrix is orthonormal = \n", xtx)

transf_x_arr = np.squeeze(np.asarray(transf_x))
df1 = pd.DataFrame(
{
    'FRAUD' : data['FRAUD'],
    'TOTAL_SPEND' : transf_x_arr[:, 0],
    'DOCTOR_VISITS' : transf_x_arr[:, 1],
    'NUM_CLAIMS' : transf_x_arr[:, 2],
    'MEMBER_DURATION' :transf_x_arr[:, 3],
    'OPTOM_PRESC' : transf_x_arr[:, 4],
    'NUM_MEMBERS' : transf_x_arr[:, 5]
})

train_data = df1[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]
y = data['FRAUD']
neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(train_data, y)

# Score value
Score_value = nbrs.score(train_data, y, sample_weight=None)
print("the score value is",Score_value)
Sample = np.mat([7500, 15, 3, 127, 2, 2])
sample_t_sample = Sample.transpose() * Sample
Sample_f = np.mat(Sample) * transf
Sample_q = np.array(Sample_f)
prediction = nbrs.predict_proba(Sample_f)
print("prediction", prediction)
distances, indices = nbrs.kneighbors(Sample_q[[0]],  n_neighbors=5)
for i in indices:
    print(data.loc[i])



