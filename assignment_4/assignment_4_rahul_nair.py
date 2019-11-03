import numpy as np
import pandas as pd
import scipy as sc
import sympy as sy
import math as myth
import itertools
import statsmodels.api as stats

#  FOR QUESTION 1
# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'Y'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

data = pd.read_csv('/Users/rahulnair/desktop/Purchase_Likelihood.csv',
                       delimiter=',')

data= data.dropna()

# Specify Origin as a categorical variable
y = data['A'].astype('category')

# Specify group_size, homeowner and married_couple as categorical variables
group_size = pd.get_dummies(data[['group_size']].astype('category'))
homeowner = pd.get_dummies(data[['homeowner']].astype('category'))
married_couple = pd.get_dummies(data[['married_couple']].astype('category'))

# Intercept only model
designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')

# Intercept + group_size
designX = stats.add_constant(group_size, prepend=True)
LLK_1g, DF_1g, fullParams_1g = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1g - LLK0)
testDF = DF_1g - DF0
testPValue = sc.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Intercept + group_size + homeowner
designX = group_size
designX = designX.join(homeowner)
designX = stats.add_constant(designX, prepend=True)
LLK_1g_1h, DF_1g_1h, fullParams_1g_1h = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1g_1h - LLK_1g)
testDF = DF_1g_1h - DF_1g
testPValue = sc.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Intercept + group_size + homeowner +married_couple
designX = group_size
designX = designX.join(homeowner)
designX = designX.join(married_couple)
designX = stats.add_constant(designX, prepend=True)
LLK_1g_1h_1m, DF_1g_1h_1m, fullParams_1g_1h_1m = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1g_1h_1m - LLK_1g_1h)
testDF = DF_1g_1h_1m - DF_1g_1h
testPValue = sc.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner
g_h = create_interaction (group_size, homeowner)
designX = group_size
designX = designX.join(homeowner)
designX = designX.join(married_couple)
designX = designX.join(g_h)
designX = stats.add_constant(designX, prepend=True)
LLK_1g_1h_1m_gh, DF_1g_1h_1m_gh, fullParams_1g_1h_1m_gh = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1g_1h_1m_gh - LLK_1g_1h_1m)
testDF = DF_1g_1h_1m_gh - DF_1g_1h_1m
testPValue = sc.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple
h_m = create_interaction (homeowner, married_couple)
designX = group_size
designX = designX.join(homeowner)
designX = designX.join(married_couple)
designX = designX.join(g_h)
designX = designX.join(h_m)
designX = stats.add_constant(designX, prepend=True)
LLK_1g_1h_1m_gh_hm, DF_1g_1h_1m_gh_hm, fullParams_1g_1h_1m_gh_hm = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1g_1h_1m_gh_hm - LLK_1g_1h_1m_gh)
testDF = DF_1g_1h_1m_gh_hm - DF_1g_1h_1m_gh
testPValue = sc.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
print("Full parameters", fullParams_1g_1h_1m_gh_hm)

# Calculating  the Feature Importance Index as the negative base-10 logarithm of the significance value
gr_log = -myth.log10(4.347870389027117e-210)
mc_log = -myth.log10(4.3064572180356084e-19)
g_h_log = -myth.log10(5.5121059685664295e-52)
h_m_log = -myth.log10(4.13804354793157e-16)

# Degree of freedom of the model
print("Degree of freedom", DF_1g_1h_1m_gh_hm)

# for calculating 16 possibilities considering the interaction terms
A_1 = fullParams_1g_1h_1m_gh_hm['0_y']
A_2 = fullParams_1g_1h_1m_gh_hm['1_y']
r = []
for i in range(1,5):
    for j in range(5,7):
        for k in range(7,9):
            sum=0
            sum = A_2[0] + A_2[i] + A_2[j] + A_2[k]
            if i==1 and j==5:
                sum+=A_2[9]
            elif i== 2 and j==5:
                sum+=A_2[11]
            elif i== 3 and j==5:
                sum+=A_2[13]
            if j==5  and k==7:
                sum+=A_2[17]
            m = myth.exp(sum)
            r.append(m)
a = []
for i in range(1,5):
    for j in range(5,7):
        for k in range(7,9):
            sum=0
            sum = A_1[0] + A_1[i] + A_1[j] + A_1[k]
            if i==1 and j==5:
                sum+=A_1[9]
            elif i== 2 and j==5:
                sum+=A_1[11]
            elif i== 3 and j==5:
                sum+=A_1[13]
            if j==5  and k==7:
                sum+=A_1[17]
            m = myth.exp(sum)
            a.append(m)
pr0 = []
pr1 = []
pr2 = []
for i in range(0,16):
    pr0.append(1/(a[i]+r[i]+1))
for i in range(0,16):
    pr1.append(a[i]*pr0[i])
for i in range(0,16):
    pr2.append(r[i]*pr0[i])
odds = []
for i in range(0,16):
    odds.append(pr1[i]/pr0[i])

# for finding the maximum odds value
print(odds.index(max(odds)))

# FOR QUESTION 2
# for calculating the frequency counts of the target variable
freq = data.groupby('A').size()

# for calculating the proportion of the target variable
prop0 = data.groupby('A').size() / data.shape[0]

# CrossTable for "group_size"
data_group_size = pd.crosstab(data.A, data.group_size, margins = False, dropna = False)
print(data_group_size)

# CrossTable for "homeowner"
data_homeowner = pd.crosstab(data.A, data.homeowner, margins = False, dropna = False)
print(data_homeowner)

# CrossTable for "married_couple"
data_married_couple = pd.crosstab(data.A, data.married_couple, margins = False, dropna = False)
print(data_married_couple)

# for finding the Cramer's V statistics
def cramerV(xCat, yCat):
    obsCount = pd.crosstab(index=xCat, columns=yCat, margins=False, dropna=True)
    cTotal = obsCount.sum(axis=1)
    rTotal = obsCount.sum(axis=0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))

    chiSqStat = ((obsCount - expCount) ** 2 / expCount).to_numpy().sum()

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)

    return (cramerV)

cramerV_group_size = cramerV(data.group_size, data.A)
print("Cramer's V Value for group_size is: ", cramerV_group_size)

cramerV_homeowner = cramerV(data.homeowner, data.A)
print("Cramer's V Value for homeowner is: ", cramerV_homeowner)

cramerV_married_couple = cramerV(data.married_couple, data.A)
print("Cramer's V Value for married_couple is: ", cramerV_married_couple)

# calculating naïve bayes for 16 possibilities
target_count = data.groupby('A').count()['group_size']
target_prop = target_count  / data.shape[0]
target_grouped = pd.DataFrame({'A': target_count.index,
                             'Count': target_count.values,
                             'Class_Probabilities': target_prop.values})


def valid_probabilities(pred):
    conditional_0 = ((target_grouped['Count'][0] / target_grouped['Count'].sum()) * 
                   (data_group_size[pred[0]][0] / data_group_size.loc[[0]].sum(axis=1)[0]) * 
                   (data_homeowner[pred[1]][0] / data_homeowner.loc[[0]].sum(axis=1)[0]) * 
                   (data_married_couple[pred[2]][0] / data_married_couple.loc[[0]].sum(axis=1)[0]))
    conditional_1 = ((target_grouped['Count'][1] / target_grouped['Count'].sum()) * 
                   (data_group_size[pred[0]][1] / data_group_size.loc[[1]].sum(axis=1)[1]) * 
                   (data_homeowner[pred[1]][1] / data_homeowner.loc[[1]].sum(axis=1)[1]) * 
                   (data_married_couple[pred[2]][1] / data_married_couple.loc[[1]].sum(axis=1)[1]))
    conditional_2 = ((target_grouped['Count'][2] / target_grouped['Count'].sum()) * 
                   (data_group_size[pred[0]][2] / data_group_size.loc[[2]].sum(axis=1)[2]) * 
                   (data_homeowner[pred[1]][2] / data_homeowner.loc[[2]].sum(axis=1)[2]) * 
                   (data_married_couple[pred[2]][2] / data_married_couple.loc[[2]].sum(axis=1)[2]))
    sum_conditionals = conditional_0 + conditional_1 + conditional_2
    valid_prob_0 = conditional_0 / sum_conditionals
    valid_prob_1 = conditional_1 / sum_conditionals
    valid_prob_2 = conditional_2 / sum_conditionals
    return [valid_prob_0, valid_prob_1, valid_prob_2]

group_sizes = sorted(list(data.group_size.unique()))
homeowners = sorted(list(data.homeowner.unique()))
married_couples = sorted(list(data.married_couple.unique()))
combinations = list(itertools.product(group_sizes, homeowners, married_couples))
naive_bayes_probabilities = []
for combination in combinations:
    temp = [valid_probabilities(combination)]
    naive_bayes_probabilities.extend(temp)
print("naïve bayes probability", naive_bayes_probabilities)
maximum=[]
for i in range(len(naive_bayes_probabilities)):
    temp=naive_bayes_probabilities[i][1]/naive_bayes_probabilities[i][0]
    maximum.append([temp])
print(np.array(maximum).max())
max_val = np.array(maximum).max()
index = np.where(maximum == max_val)[0][0]
print("The maximum value occurs when group_size, homeowner, married_couple values are: ",combinations[index])
print("The maximum value is: ", max_val)
