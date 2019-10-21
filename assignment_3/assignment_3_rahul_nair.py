import matplotlib.pyplot as plt
import numpy
import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import math
import itertools

train_split = 0.7
test_split = 0.3
claim_history = pd.read_csv('/Users/rahulnair/desktop/claim_history.csv',delimiter=',')
train_data, test_data = train_test_split(claim_history, test_size = test_split, random_state = 27513, stratify = claim_history["CAR_USE"])

# FOR QUESTION-1(a)
print(train_data.groupby('CAR_USE').size())
print("Training data Size: ",train_data.shape[0])
# print(train_data.groupby('CAR_USE').size()/train_data.shape[0])
prob_comm_train = train_data.groupby('CAR_USE').size()[0]/train_data.shape[0]
prob_pri_train = train_data.groupby('CAR_USE').size()[1]/train_data.shape[0]
print("Proportion of commercial vehicles in training data: ",prob_comm_train)
print("Proportion of private vehicles in training data: ",prob_pri_train)

count_comm = train_data.groupby('CAR_USE').size()[0]
count_pri = train_data.groupby('CAR_USE').size()[1]

train_comm = train_data.groupby('CAR_USE')

# FOR QUESTION-1(b)
print(test_data.groupby('CAR_USE').size())
print("Testing data Size: ", test_data.shape[0])
# print(test_data.groupby('CAR_USE').size()/test_data.shape[0])
prob_comm_test = test_data.groupby('CAR_USE').size()[0]/test_data.shape[0]
prob_pri_test = test_data.groupby('CAR_USE').size()[1]/test_data.shape[0]
print("Proportion of commercial vehicles in testing data: ", prob_comm_test)
print("Proportion of private vehicles in testing data: ",prob_pri_test)

# FOR QUESTION-1(c)
prob_train_comm = (prob_comm_train*train_split)/((prob_comm_train*train_split)+(prob_comm_test*test_split))
print("probability that an observation is in the Training partition given that 'CAR_USE = Commercial':- ", prob_train_comm)

# FOR QUESTION-1(d)
prob_test_pri = (prob_pri_test*test_split)/((prob_pri_train*train_split)+(prob_pri_test*test_split))
print("probability that an observation is in the Test partition given that 'CAR_USE = Private':- ", prob_test_pri)

#  predictor variables
X_name = ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']

# target variable
Y_name = 'CAR_USE'


def entropy(count_comm,count_pri):
    type_comm = ((count_comm)/(count_comm+count_pri))
    type_pri = ((count_pri)/(count_comm+count_pri))
    if ( type_comm>0 and type_pri>0):
        entropy_class_CAR_USE = -(type_comm*math.log(type_comm,2)) - (type_pri*math.log(type_pri,2))
        return(entropy_class_CAR_USE)
    else:
        return(999)


# FOR QUESTION-2(a)
entropy_CAR_USE = entropy(count_comm,count_pri)
print('ENTROPY OF ROOT NODE: ', entropy_CAR_USE)

TrainData = train_data[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']]
for_car_type = TrainData.groupby(['CAR_TYPE','CAR_USE']).size()
minivan_0 = for_car_type['Minivan'][0]
minivan_1 = for_car_type['Minivan'][1]
panel_truck_0 = for_car_type['Panel Truck'][0]
panel_truck_1 = 0
pickup_0 = for_car_type['Pickup'][0]
pickup_1 = for_car_type['Pickup'][1]
suv_0 = for_car_type['SUV'][0]
suv_1 = for_car_type['SUV'][1]
sports_0 = for_car_type['Sports Car'][0]
sports_1 = for_car_type['Sports Car'][1]
van_0 = for_car_type['Van'][0]
van_1 = for_car_type['Van'][1]
car_dict = {'Minivan': [minivan_0, minivan_1], 'Panel Truck': [panel_truck_0, panel_truck_1],'Pickup': [pickup_0 , pickup_1], 'SUV': [suv_0, suv_1], 'Sports Car': [sports_0, sports_1], 'Van': [van_0, van_1]}
car_dict_keys = list(car_dict.keys())
car_types = list(train_data['CAR_TYPE'].unique())


def create_proper_subset(car_types):
    all_car_types = []
    for car in range(1,len(car_types)):
        all_car_types.extend(itertools.combinations(car_types,car))
    proper_subset_caruse = []
    a = (len(car_types))
    for value in all_car_types:
        b = len(value)
        l_to_find = a - b
        for val in all_car_types:
            if len(val) == l_to_find:
                set_value = set(value)
                set_val = set(val)
                if(len(set_val.intersection(set_value)) == 0):
                    check_dupe = [(val), (value)]
                    if check_dupe not in proper_subset_caruse:
                        t1 = [(value),(val)]
                        proper_subset_caruse.append(t1)
    return(proper_subset_caruse)

proper_subset_caruse = create_proper_subset(car_types)


def min_entropy(dict_, proper_subset):
    split_entropy=[]
    i=0
    min_ = 999
    index = -1
    for s in proper_subset:
        xcount_0 = 0
        xcount_1 = 0
        ycount_0 = 0
        ycount_1 = 0
        entropy_s = 0
        print(s[0],s[1])
        for x in s[0]:
            xcount_0 = xcount_0 + dict_.get(x)[0]
            xcount_1 = xcount_1 + dict_.get(x)[1]
        for y in s[1]:
            ycount_0 = ycount_0 + dict_.get(y)[0]
            ycount_1 = ycount_1 + dict_.get(y)[1]
        x_entropy = entropy(xcount_0,xcount_1)
        y_entropy = entropy(ycount_0,ycount_1)
        x_count = xcount_0 + xcount_1
        y_count = ycount_0 + ycount_1
        total_count = x_count + y_count
        s_entropy = ((x_count/total_count)*x_entropy) + ((y_count/total_count)*y_entropy)
        print(s_entropy)
        split_entropy.append(s_entropy)
        if (min_ > min(split_entropy)):
            min_ = min(split_entropy)
            index = s
#     print(index)
    return(index, min_)
split_car_type, min_set_CAR_TYPE = min_entropy(car_dict, proper_subset_caruse)


print(min_set_CAR_TYPE)
print(split_car_type)

for_occupation = TrainData.groupby(['OCCUPATION','CAR_USE']).size()
blue_collar_0 = for_occupation['Blue Collar'][0]
blue_collar_1 = for_occupation['Blue Collar'][1]
Clerical_0 = for_occupation['Clerical'][0]
Clerical_1 = for_occupation['Clerical'][1]
Doctor_0 = 0
Doctor_1 = for_occupation['Doctor'][0]
Home_Maker_0 = for_occupation['Home Maker'][0]
Home_Maker_1 = for_occupation['Home Maker'][1]
Lawyer_0 = 0
Lawyer_1= for_occupation['Lawyer'][0]
Manager_0 = for_occupation['Manager'][0]
Manager_1 = for_occupation['Manager'][1]
Professional_0 = for_occupation['Professional'][0]
Professional_1 = for_occupation['Professional'][1]
Student_0 = for_occupation['Student'][0]
Student_1 = for_occupation['Student'][1]
Unknown_0 = for_occupation['Unknown'][0]
Unknown_1 = for_occupation['Unknown'][1]
occ_dict = {'Blue Collar': [blue_collar_0, blue_collar_1], 'Clerical': [Clerical_0, Clerical_1],'Doctor': [Doctor_0 , Doctor_1], 'Home Maker': [Home_Maker_0, Home_Maker_1], 'Lawyer': [Lawyer_0, Lawyer_1], 'Manager': [Manager_0, Manager_1], 'Professional': [Professional_0, Professional_1], 'Student': [Student_0, Student_1], 'Unknown': [Unknown_0, Unknown_1]}
occ_types = sorted(list(train_data['OCCUPATION'].unique()))
all_occ_types = []
for occ in range(1,len(occ_types)):
    all_occ_types.extend(itertools.combinations(occ_types,occ))
all_occ_types
proper_subset_occ = []
a = (len(occ_types))
for value in all_occ_types:
    b = len(value)
    l_to_find = a - b
    for val in all_occ_types:
        if len(val) == l_to_find:
            set_value = set(value)
            set_val = set(val)
            if(len(set_val.intersection(set_value)) == 0):
                check_dupe = [(val), (value)]
                if check_dupe not in proper_subset_occ:
                    t1 = [(value),(val)]
                    proper_subset_occ.append(t1)


split_occ, min_set_OCCUPATION = min_entropy(occ_dict, proper_subset_occ)
print(min_set_OCCUPATION)
print(split_occ)

for_education = TrainData.groupby(['EDUCATION','CAR_USE']).size()
for_education
ordered_ordinal_predictors = ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors']
proper_subset_edu = []
for predictor in range(1,len(ordered_ordinal_predictors)):
    first = tuple(ordered_ordinal_predictors[:predictor])
    last = tuple(ordered_ordinal_predictors[predictor:])
    merged = [(first),(last)]
    proper_subset_edu.append(merged)

Bachelors_0 = for_education['Bachelors'][0]
Bachelors_1 = for_education['Bachelors'][1]
Below_High_School_0 = for_education['Below High School'][0]
Below_High_School_1 = for_education['Below High School'][1]
Doctors_0 = for_education['Doctors'][0]
Doctors_1 = for_education['Doctors'][1]
High_School_0 = for_education['High School'][0]
High_School_1 = for_education['High School'][1]
Masters_0 = for_education['Masters'][0]
Masters_1= for_education['Masters'][1]
edu_dict = {'Below High School': [Below_High_School_0, Below_High_School_1], 'High School': [High_School_0, High_School_1], 'Bachelors': [Bachelors_0, Bachelors_1], 'Masters': [Masters_0, Masters_1], 'Doctors': [Doctors_0 , Doctors_1],}
split_edu, min_set_EDUCATION = min_entropy(edu_dict, proper_subset_edu)
print(min_set_EDUCATION)
print(split_edu)
least_entropy = [min_set_CAR_TYPE,min_set_EDUCATION,min_set_OCCUPATION]

# the minimumum entropy at first layer
print("Mninimum entropy is :", min_set_OCCUPATION)

# the split at first layer
print("the split for first layer: ", split_occ)

# count of training data at first layer
print("total count of training data at first layer: ", train_data['CAR_USE'].count())

# count of 'CAR_USE == Commercial'
print("total count of commercial vehicles at first layer: ", train_data[train_data.CAR_USE == 'Commercial']['CAR_USE'].count())

# count of 'CAR_USE == Private'
print("total count of private vehicles at first layer: ", train_data[train_data.CAR_USE == 'Private']['CAR_USE'].count())

occ_splitset_1 = ['Blue Collar', 'Student', 'Unknown']
proper_set_occ_1 = create_proper_subset(occ_splitset_1)
occ1_train =  TrainData[TrainData.OCCUPATION.isin(occ_splitset_1)]
occ1_edu = occ1_train.groupby(['EDUCATION','CAR_USE']).size()
occ1_edu_Bachelors_0 = occ1_edu['Bachelors'][0]
occ1_edu_Bachelors_1 = occ1_edu['Bachelors'][1]
occ1_edu_Below_High_School_0 = occ1_edu['Below High School'][0]
occ1_edu_Below_High_School_1 = occ1_edu['Below High School'][1]
occ1_edu_Doctors_0 = occ1_edu['Doctors'][0]
occ1_edu_Doctors_1 = occ1_edu['Doctors'][1]
occ1_edu_High_School_0 = occ1_edu['High School'][0]
occ1_edu_High_School_1 = occ1_edu['High School'][1]
occ1_edu_Masters_0 = occ1_edu['Masters'][0]
occ1_edu_Masters_1= occ1_edu['Masters'][1]

occ1_edu_dict = {'Below High School': [occ1_edu_Below_High_School_0, occ1_edu_Below_High_School_1], 'High School': [occ1_edu_High_School_0, occ1_edu_High_School_1], 'Bachelors': [occ1_edu_Bachelors_0, occ1_edu_Bachelors_1], 'Masters': [occ1_edu_Masters_0, occ1_edu_Masters_1], 'Doctors': [occ1_edu_Doctors_0 , occ1_edu_Doctors_1],}
occ1_edu_split, min_occ1_edu_ = min_entropy(occ1_edu_dict,proper_subset_edu)
print(min_occ1_edu_)
print(occ1_edu_split)
occ1_cartype = occ1_train.groupby(['CAR_TYPE','CAR_USE']).size()
occ1_minivan_0 = occ1_cartype['Minivan'][0]
occ1_minivan_1 = occ1_cartype['Minivan'][1]
occ1_panel_truck_0 = occ1_cartype['Panel Truck'][0]
occ1_panel_truck_1 = 0
occ1_pickup_0 = occ1_cartype['Pickup'][0]
occ1_pickup_1 = occ1_cartype['Pickup'][1]
occ1_suv_0 = occ1_cartype['SUV'][0]
occ1_suv_1 = occ1_cartype['SUV'][1]
occ1_sports_0 = occ1_cartype['Sports Car'][0]
occ1_sports_1 = occ1_cartype['Sports Car'][1]
occ1_van_0 = occ1_cartype['Van'][0]
occ1_van_1 = occ1_cartype['Van'][1]
occ1_car_dict = {'Minivan': [occ1_minivan_0, occ1_minivan_1], 'Panel Truck': [occ1_panel_truck_0, occ1_panel_truck_1],'Pickup': [occ1_pickup_0 , occ1_pickup_1], 'SUV': [occ1_suv_0, occ1_suv_1], 'Sports Car': [occ1_sports_0, occ1_sports_1], 'Van': [occ1_van_0, occ1_van_1]}
occ1_cartype, min_occ1_cartype = min_entropy(occ1_car_dict,proper_subset_caruse)
print(min_occ1_cartype)
print(occ1_cartype)
list_entropy_second_layer_left = [min_occ1_edu_, min_occ1_cartype]

# the minimumum entropy for left part of second layer
print("Mninimum entropy for left part of second layer is :", min_occ1_edu_)

# the split
print("the split for left part of second layer: ", occ1_edu_split)

# count of training data at left part of second layer
print("total count of training data at second layer (left): ", occ1_train['CAR_USE'].count())

# count of 'CAR_USE == Commercial'
print("total count of commercial vehicles at left part of second layer: ", occ1_train[occ1_train.CAR_USE == 'Commercial']['CAR_USE'].count())

# count of 'CAR_USE == Private'
print("total count of private vehicles at left part of second layer: ", occ1_train[occ1_train.CAR_USE == 'Private']['CAR_USE'].count())

occ_splitset_2 = ['Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional']
proper_set_occ_2 = create_proper_subset(occ_splitset_2)
occ2_train =  TrainData[TrainData.OCCUPATION.isin(occ_splitset_2)]
occ2_edu = occ2_train.groupby(['EDUCATION','CAR_USE']).size()
occ2_edu_Bachelors_0 = occ2_edu['Bachelors'][0]
occ2_edu_Bachelors_1 = occ2_edu['Bachelors'][1]
occ2_edu_Below_High_School_0 = occ2_edu['Below High School'][0]
occ2_edu_Below_High_School_1 = occ2_edu['Below High School'][1]
occ2_edu_Doctors_0 = occ2_edu['Doctors'][0]
occ2_edu_Doctors_1 = occ2_edu['Doctors'][1]
occ2_edu_High_School_0 = occ2_edu['High School'][0]
occ2_edu_High_School_1 = occ2_edu['High School'][1]
occ2_edu_Masters_0 = occ2_edu['Masters'][0]
occ2_edu_Masters_1= occ2_edu['Masters'][1]
occ2_edu_dict = {'Below High School': [occ2_edu_Below_High_School_0, occ2_edu_Below_High_School_1], 'High School': [occ2_edu_High_School_0, occ2_edu_High_School_1], 'Bachelors': [occ2_edu_Bachelors_0, occ2_edu_Bachelors_1], 'Masters': [occ2_edu_Masters_0, occ2_edu_Masters_1], 'Doctors': [occ2_edu_Doctors_0 , occ2_edu_Doctors_1],}
occ2_edu, min_occ2_edu_ = min_entropy(occ2_edu_dict,proper_subset_edu)
print(min_occ2_edu_)
print(occ2_edu)
occ2_cartype = occ2_train.groupby(['CAR_TYPE','CAR_USE']).size()
occ2_minivan_0 = occ2_cartype['Minivan'][0]
occ2_minivan_1 = occ2_cartype['Minivan'][1]
occ2_panel_truck_0 = occ2_cartype['Panel Truck'][0]
occ2_panel_truck_1 = 0
occ2_pickup_0 = occ2_cartype['Pickup'][0]
occ2_pickup_1 = occ2_cartype['Pickup'][1]
occ2_suv_0 = occ2_cartype['SUV'][0]
occ2_suv_1 = occ2_cartype['SUV'][1]
occ2_sports_0 = occ2_cartype['Sports Car'][0]
occ2_sports_1 = occ2_cartype['Sports Car'][1]
occ2_van_0 = occ2_cartype['Van'][0]
occ2_van_1 = occ2_cartype['Van'][1]
occ2_car_dict = {'Minivan': [occ2_minivan_0, occ2_minivan_1], 'Panel Truck': [occ2_panel_truck_0, occ2_panel_truck_1],'Pickup': [occ2_pickup_0 , occ2_pickup_1], 'SUV': [occ2_suv_0, occ2_suv_1], 'Sports Car': [occ2_sports_0, occ2_sports_1], 'Van': [occ2_van_0, occ2_van_1]}
occ2_cartype, min_occ2_cartype = min_entropy(occ2_car_dict,proper_subset_caruse)
print(min_occ2_cartype)
print(occ2_cartype)
list_entropy_second_layer_right = [min_occ2_edu_, min_occ2_cartype]

# the minimumum entropy for right part of second layer
print("Mninimum entropy for right part of second layer is :", min_occ2_cartype)

# the split
print("the split for right part of second layer: ", occ2_cartype)

# count of training data at right part of second layer
print("total count of training data at second layer (right): ", occ2_train['CAR_USE'].count())

# count of 'CAR_USE == Commercial'
print("total count of commercial vehicles at right part of second layer: ", occ2_train[occ2_train.CAR_USE == 'Commercial']['CAR_USE'].count())

# count of 'CAR_USE == Private'
print("total count of private vehicles at right part of second layer: ", occ2_train[occ2_train.CAR_USE == 'Private']['CAR_USE'].count())

# for left leaf of second layer (left)
qwerty = [('Below High School',), ('High School', 'Bachelors', 'Masters', 'Doctors')]
qwe = occ1_train[occ1_train.EDUCATION.isin(qwerty[0])]
qwe.count()

q = qwe[qwe.CAR_USE == 'Commercial']['CAR_USE'].count()
q1 = qwe[qwe.CAR_USE == 'Private']['CAR_USE'].count()

# for right leaf of second layer (left)
qwe1 = occ1_train[occ1_train.EDUCATION.isin(qwerty[1])]
qwe1.count()

r = qwe1[qwe1.CAR_USE == 'Commercial']['CAR_USE'].count()
r1 = qwe1[qwe1.CAR_USE == 'Private']['CAR_USE'].count()

# for left leaf of second layer (right)
asd = [('Minivan', 'SUV', 'Sports Car'), ('Van', 'Panel Truck', 'Pickup')]
zac = occ2_train[occ2_train.CAR_TYPE.isin(asd[0])]
zac.count()

# for right leaf of second layer (right)
zac1 = occ2_train[occ2_train.CAR_TYPE.isin(asd[1])]
zac1.count()
e = zac[zac.CAR_USE == 'Commercial']['CAR_USE'].count()
e1 = zac[zac.CAR_USE == 'Private']['CAR_USE'].count()
t = zac1[zac1.CAR_USE == 'Commercial']['CAR_USE'].count()
t1 = zac1[zac1.CAR_USE == 'Private']['CAR_USE'].count()

TestData = test_data[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']]
print(TestData)
occ = [('Blue Collar', 'Student', 'Unknown'), ('Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional')]
occ_entropy = 0.7148805225259209
temp1 = TestData[TestData.OCCUPATION.isin(occ[0])]
edu = [('Below High School',), ('High School', 'Bachelors', 'Masters', 'Doctors')]
edu_entropy = 0.9343298080392604
df1 = temp1[temp1.EDUCATION.isin(edu[0])]
print(df1)
grouped1 = df1.groupby(['EDUCATION','CAR_USE']).size()
print(grouped1)
df1_comm = len(df1[df1['CAR_USE'] == 'Commercial'])
df1_pri = len(df1[df1['CAR_USE'] == 'Private'])
df1_ep = df1_comm/(df1_comm + df1_pri)
print(df1_comm,df1_pri,df1_ep)
df1['EP'] = df1_ep
df2 = temp1[temp1.EDUCATION.isin(edu[1])]
print(df2)
grouped2 = df2.groupby(['EDUCATION','CAR_USE']).size()
print(grouped2)
df2_comm = len(df2[df2['CAR_USE'] == 'Commercial'])
df2_pri = len(df2[df2['CAR_USE'] == 'Private'])
df2_ep = df2_comm/(df2_comm + df2_pri)
print(df2_comm,df2_pri,df2_ep)
df2['EP'] = df2_ep
temp2 = TestData[TestData.OCCUPATION.isin(occ[1])]
cartype = [('Minivan', 'SUV', 'Sports Car'), ('Van', 'Panel Truck', 'Pickup')]
cartype_entropy = 0.7573352263531923
df3 = temp2[temp2.CAR_TYPE.isin(cartype[0])]
print(df3)
grouped3 = df3.groupby(['CAR_TYPE','CAR_USE']).size()
print(grouped3)
df3_comm = len(df3[df3['CAR_USE'] == 'Commercial'])
df3_pri = len(df3[df3['CAR_USE'] == 'Private'])
df3_ep = df3_comm/(df3_comm + df3_pri)
print(df3_comm,df3_pri,df3_ep)
df3['EP'] = df3_ep
df4 = temp2[temp2.CAR_TYPE.isin(cartype[1])]
print(df4)
grouped4 = df4.groupby(['CAR_TYPE','CAR_USE']).size()
print(grouped4)
df4_comm = len(df4[df4['CAR_USE'] == 'Commercial'])
df4_pri = len(df4[df4['CAR_USE'] == 'Private'])
df4_ep = df4_comm/(df4_comm + df4_pri)
print(df4_comm,df4_pri,df4_ep)
df4['EP'] = df4_ep
merged = pd.concat([df1,df2,df3,df4])
Y = merged['CAR_USE'].valuespredProbY = merged['EP'].values
nY = Y.shape[0]
predProbY = merged['EP'].values

# Determine the predicted class of Y
predY = numpy.empty_like(Y)
for i in range(nY):
    if (predProbY[i] > 0.368049):
        predY[i] = 'Commercial'
    else:
        predY[i] = 'Private'

# Calculate the Root Average Squared Error
RASE = 0.0
for i in range(nY):
    if (Y[i] == 'Commercial'):
        RASE += (1 - predProbY[i])**2
    else:
        RASE += (0 - predProbY[i])**2
RASE = numpy.sqrt(RASE/nY)

# Calculate the Root Mean Squared Error
Y_true = 1.0 * numpy.isin(Y, ['Commercial'])
RMSE = metrics.mean_squared_error(Y_true, predProbY)
RMSE = numpy.sqrt(RMSE)

# For binary y_true, y_score is supposed to be the score of the class with greater label.
AUC = metrics.roc_auc_score(Y_true, predProbY)
accuracy = metrics.accuracy_score(Y, predY)

print('                  Accuracy: {:.13f}' .format(accuracy))
print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
print('          Area Under Curve: {:.13f}' .format(AUC))
print('Root Average Squared Error: {:.13f}' .format(RASE))
print('   Root Mean Squared Error: {:.13f}' .format(RMSE))

# FOR QUESTION-3
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

