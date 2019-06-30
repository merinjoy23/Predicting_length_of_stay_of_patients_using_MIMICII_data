#!/usr/bin/env python
# coding: utf-8

# ## Predicting the Length of Stay (LOS) in hospitals using the MIMIC III dataset

# In[6]:


# Import the required tools
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV


# In[7]:


# Connect to sql
conn = sqlite3.connect('/Users/Merin/Desktop/HAP880/HAP_880_Project/mimic.db/mimic.db')


# In[8]:


# View the tables in MIMIC database
tables = pd.read_sql('select * from sqlite_master', conn)
tables


# In[9]:


# Tables selected for analysis

# primary admissions information
df_admin = pd.read_sql('select * from admissions', con=conn)

# information on nature of patients condition
df_cpt = pd.read_sql('select * from cptevents', con=conn)

# Intensive Care Unit (ICU) details for each admission to hospital
df_icustays = pd.read_sql('select * from icustays', con=conn)

# services used by the patient
df_services = pd.read_sql('select * from services', con=conn)

# demographic information on patients
df_patients = pd.read_sql('select * from patients', con=conn)

# information on diagonsis
df_diagnoses = pd.read_sql('select * from diagnoses_icd', con=conn)

# information on procedures
df_procedures = pd.read_sql('select * from procedures_icd', con=conn)


# ## Admissions table

# In[10]:


df_admin.info()


# In[11]:


df_admin.head()


# In[12]:


print(" Admission contains {} unique admission events".format(df_admin['HADM_ID'].nunique()))
print(" Admission contains {} unique patients".format(df_admin['SUBJECT_ID'].nunique()))


# ### Length of Stay (Dependent variable)

# In[13]:


# Converting admission time and discharge time to datetime type
df_admin['ADMITTIME'] = pd.to_datetime(df_admin['ADMITTIME'])
df_admin['DISCHTIME'] = pd.to_datetime(df_admin['DISCHTIME'])

# Calculating Length of Stay (in days)
df_admin['LOS'] = (df_admin['DISCHTIME']-df_admin['ADMITTIME']).dt.total_seconds()/86400


# In[14]:


df_admin[['ADMITTIME','DISCHTIME','LOS']].head()


# In[15]:


df_admin['LOS'].describe()


# In[16]:


# In the case of death, we get a negative LOS value. We can see that the Hospital expire flag is 1 in case of a negative value 
# of LOS, which means that the patient dies in hospital
df_admin[df_admin['LOS']<0].head()


# In[17]:


# 98 rows having deaths need to be removed
df_admin[df_admin['LOS']<0].describe()


# In[18]:


# Droping rows having negative LOS, since its given time of death is before admit time
df_admin = df_admin[df_admin['LOS']>0]


# In[19]:


# Ploting a histogram of LOS
plt.hist(df_admin['LOS'],bins = 200, edgecolor='black',color ='#2388a6')
plt.xlim(0,50)
plt.title("LOS distribution for all hospital admissions")
plt.xlabel("Length of Stay (days)")
plt.ylabel("Count")
plt.show()


# ### Hospital Expire Flag

# In[20]:


# Also, we can see that 5774 admission events result in death
df_admin['LOS'].loc[df_admin['HOSPITAL_EXPIRE_FLAG'] == '1'].describe()


# ### Ethnicity

# In[21]:


# Ethnic groups in the dataset
df_admin['ETHNICITY'].value_counts()


# In[22]:


# Reduce ethnicity into generic categories
df_admin['ETHNICITY'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
df_admin['ETHNICITY'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
df_admin['ETHNICITY'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
df_admin['ETHNICITY'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
df_admin['ETHNICITY'].replace(['UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER', 
                         'UNKNOWN/NOT SPECIFIED'], value='UNKNOWN', inplace=True)
df_admin['ETHNICITY'].replace(['MULTI RACE ETHNICITY', 'PORTUGUESE','AMERICAN INDIAN/ALASKA NATIVE',
                               'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','CARIBBEAN ISLAND','SOUTH AMERICAN',
                               'MIDDLE EASTERN','AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE'], 
                              value='OTHER', inplace=True)
df_admin['ETHNICITY'].value_counts()


# In[23]:


# Boxplot function for LOS with categories
def boxplot_los_groupby(var, los_range=(-1, 30), size=(8,4)):
    result = df_admin[[var, 'LOS']].groupby(var).median().reset_index()
    category = result[var].values.tolist()
    box_data = []
    for cat in category:
        box_data.append(df_admin['LOS'].loc[df_admin[var]==cat].values)
    fig, ax = plt.subplots(figsize=size)
    fig.canvas.draw()
    ax.boxplot(box_data, 0, '', vert=False)
    ax.set_xlim(los_range)
    ax.set_yticklabels(category)
    ax.set_xlabel('Length of Stay (days)') 
    ax.set_title('Comparison of {} categories'.format(var))
    plt.show();
    
boxplot_los_groupby('ETHNICITY', los_range=(-1, 30))


# ### Admission Type

# In[24]:


# Admission type groups
df_admin['ADMISSION_TYPE'].value_counts()


# In[25]:


# We can see that Urgent have maximum median LOS
boxplot_los_groupby('ADMISSION_TYPE', los_range=(-1, 35), size =(7,4))


# ### Insurance

# In[26]:


# Insurance categories
df_admin['INSURANCE'].value_counts()


# In[27]:


# We can see that Self Pay has the least median LOS
boxplot_los_groupby('INSURANCE', los_range=(-1, 30), size =(7,4))


# ### Marital Status

# In[28]:


# Marital Status categories. NaN has 10097 counts
df_admin['MARITAL_STATUS'].value_counts()


# In[29]:


# Placing blanks under UNKNOWN (DEFAULT)
df_admin['MARITAL_STATUS'].replace([''], value='UNKNOWN (DEFAULT)', inplace=True)
df_admin['MARITAL_STATUS'].value_counts()


# In[30]:


# Marital status with LOS
boxplot_los_groupby('MARITAL_STATUS', los_range=(-1, 30), size =(7,4))


# ### Admission Location

# In[31]:


df_admin['ADMISSION_LOCATION'].value_counts()


# In[32]:


# Replacing ** INFO NOT AVAILABLE ** to UNKNOWN
df_admin['ADMISSION_LOCATION'].replace(['** INFO NOT AVAILABLE **'], value='UNKNOWN', inplace=True)
df_admin['ADMISSION_LOCATION'].value_counts()


# In[33]:


# Maximum median LOS is for transfer from other heath centers
boxplot_los_groupby('ADMISSION_LOCATION', los_range=(-1, 50), size =(7,5))


# ### Discharge Location

# In[34]:


df_admin['DISCHARGE_LOCATION'].value_counts()


# In[35]:


boxplot_los_groupby('DISCHARGE_LOCATION', los_range=(-1, 45), size =(7,7))


# ## Preprocessing Admissions table

# In[36]:


# Create dummy columns for categorical variables
prefix_cols = ['ADM', 'AL', 'DL', 'INS', 'MAR','ETH']
dummy_cols = ['ADMISSION_TYPE','ADMISSION_LOCATION','DISCHARGE_LOCATION',
              'INSURANCE', 'MARITAL_STATUS','ETHNICITY']
df_admin_new = pd.get_dummies(df_admin, prefix=prefix_cols, columns=dummy_cols)

# Drop unused or no longer needed columns
# Removing HOSPITAL_EXPIRE_FLAG because DL_DEAD/EXPIRED contains the same values
df_admin_new.drop(columns=['ROW_ID','DISCHTIME','DEATHTIME','LANGUAGE','RELIGION','EDREGTIME',
                           'EDOUTTIME','DIAGNOSIS','HOSPITAL_EXPIRE_FLAG','HAS_CHARTEVENTS_DATA'], inplace=True)

df_admin_new.head()


# In[37]:


df_admin_new.info()


# ## CPT events table

# In[38]:


df_cpt.info()


# In[39]:


df_cpt.head()


# ### Cost Center
# #### COSTCENTER is the cost center which billed for the corresponding CPT codes. There are two possible cost centers: ‘ICU’ and ‘Resp’. ‘Resp’ codes correspond to mechanical or non-invasive ventilation and were billed by the respiratory therapist. ‘ICU’ codes correspond to the procedures billed for by the ICU.

# In[40]:


df_cpt['COSTCENTER'].value_counts()


# ### CPT Sequence Number

# In[41]:


cpt_list1 = []
cpt_list1 = pd.DataFrame(df_cpt.groupby(['HADM_ID'], sort=False)['CPT_CD'].count()).reset_index()
cpt_list1.head()


# ## Preprocessing CPT events table

# In[42]:


cpt_list2 = []
cpt_list3 = []
cpt_list2 = df_cpt.groupby('HADM_ID')['COSTCENTER'].apply(list).reset_index()
cpt_list3 = pd.get_dummies(cpt_list2['COSTCENTER'].apply(pd.Series).stack()).max(level=0)
cpt_list3 = cpt_list3.join(cpt_list2['HADM_ID'], how="outer")
cpt_list3.head()


# In[43]:


df_cpt_new = cpt_list3.merge(cpt_list1, how='inner', on='HADM_ID')
df_cpt_new.columns = ['CPT_ICU','CPT_RESP','HADM_ID','CPT_SEQUENCE_NUMBER']
df_cpt_new.head()


# In[44]:


df_cpt_new.info()


# ## ICU stays table

# In[45]:


df_icustays.info()


# In[46]:


df_icustays.head()


# ### First care unit

# In[47]:


df_icustays['FIRST_CAREUNIT'].value_counts()


# ### Last care unit

# In[48]:


df_icustays['LAST_CAREUNIT'].value_counts()


# ## Preprocessing ICU stays table

# In[49]:


icu_list1 = []
icu_list2 = []
icu_list1 = pd.DataFrame(df_icustays.groupby('HADM_ID')['FIRST_CAREUNIT'].apply(list).reset_index())
icu_list2 = pd.get_dummies(icu_list1['FIRST_CAREUNIT'].apply(pd.Series).stack()).max(level=0)
icu_list2.head()


# In[50]:


icu_list3 = icu_list2.join(icu_list1['HADM_ID'], how="outer")
icu_list3.columns = ['FC_CCU','FC_CSRU','FC_MICU','FC_NICU','FC_SICU','FC_TSICU','HADM_ID']
icu_list3.head()


# In[51]:


icu_list4 = []
icu_list5 = []
icu_list4 = pd.DataFrame(df_icustays.groupby('HADM_ID')['LAST_CAREUNIT'].apply(list).reset_index())
icu_list5 = pd.get_dummies(icu_list4['LAST_CAREUNIT'].apply(pd.Series).stack()).max(level=0)
icu_list5.head()


# In[52]:


icu_list6 = icu_list5.join(icu_list4['HADM_ID'], how="outer")
icu_list6.columns = ['LC_CCU','LC_CSRU','LC_MICU','LC_NICU','LC_SICU','LC_TSICU','HADM_ID']
icu_list6.head()


# In[53]:


df_icustays_new = icu_list3.merge(icu_list6, how='inner', on='HADM_ID')
df_icustays_new.head()


# In[54]:


df_icustays_new.info()


# ## Services table

# In[55]:


df_services.info()


# In[56]:


df_services.head()


# ### Current Service

# In[57]:


df_services['CURR_SERVICE'].value_counts()


# ## Preprocessing Services table

# In[58]:


ser_list1 = []
ser_list2 = []
ser_list1 = pd.DataFrame(df_services.groupby('HADM_ID')['CURR_SERVICE'].apply(list).reset_index())
ser_list2 = pd.get_dummies(ser_list1['CURR_SERVICE'].apply(pd.Series).stack()).max(level=0)
ser_list2.head()


# In[59]:


df_services_new = ser_list2.join(ser_list1['HADM_ID'], how="outer")
df_services_new.head()


# In[60]:


df_services_new.info()


# ## Patients table

# In[61]:


df_patients.info()


# In[62]:


df_patients.head()


# ### Gender

# In[63]:


df_patients['GENDER'].value_counts()


# ### DOB

# In[64]:


# Convert to datetime type
df_patients['DOB'] = pd.to_datetime(df_patients['DOB'])
df_patients['DOB'].head()


# ## Preprocessing Patients table

# In[65]:


pat_list1 = []
pat_list2 = []
pat_list1 = pd.DataFrame(df_patients.groupby('SUBJECT_ID')['GENDER'].apply(list).reset_index())
pat_list2 = pd.get_dummies(pat_list1['GENDER'].apply(pd.Series).stack()).max(level=0)
pat_list2.head()


# In[66]:


pat_list3 = pat_list2.join(pat_list1['SUBJECT_ID'], how="outer")
pat_list3.head()


# In[67]:


df_patients_new = pat_list3.join(df_patients['DOB'], how="outer")
df_patients_new.head()


# In[68]:


df_patients_new.info()


# ## Procedures ICD table

# In[69]:


df_procedures.info()


# In[70]:


df_procedures.head()


# ## Preprocessing Procedures ICD table
# ### Procedure Number

# In[71]:


# “Procedures number” is used to reﬂect the total procedures performed on the patient during the stay at the ICU, without 
# focusing on the speciﬁc type of those procedures.

df_procedures_new = []
df_procedures_new = pd.DataFrame(df_procedures.groupby(['HADM_ID'], sort=False)['SEQ_NUM'].max()).reset_index()
df_procedures_new.columns =['HADM_ID','PRO_NUM']
df_procedures_new.head()


# In[72]:


df_procedures_new.info()


# ## Diagnosis ICD table

# In[73]:


df_diagnoses.head()


# In[74]:


df_diagnoses.info()


# ### ICD9_CODE

# In[75]:


df_diagnoses['ICD9_CODE'].value_counts()


# ## Preprocessing Diagnoses ICD table

# In[76]:


# Truncate first three character values. Take out codes starting with V and E
df_diagnoses['code'] = df_diagnoses['ICD9_CODE']
df_diagnoses['code'] = df_diagnoses['code'][~df_diagnoses['code'].str.contains("[a-zA-Z]").fillna(False)]
df_diagnoses['code'] = df_diagnoses['code'].replace('',np.nan, regex=True)
df_diagnoses['code'].fillna(value='999', inplace=True)


# In[77]:


df_diagnoses.head()


# In[78]:


df_diagnoses['code'] = df_diagnoses['code'].str.slice(start=0, stop=3, step=1)


# In[79]:


df_diagnoses.head()


# In[80]:


df_diagnoses['code'] = df_diagnoses['code'].astype(int)


# In[81]:


# ICD9_CODES Main Category ranges (Super categories of ICD codes)
icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), (320, 390), 
               (390, 460), (460, 520), (520, 580), (580, 630), (630, 680), (680, 710),
               (710, 740), (740, 760), (760, 780), (780, 800), (800, 1000), (1000, 2000)]

# Category names
diagnoses_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood_diseases',
             4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
             8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin', 
             12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'symptoms',
             16: 'injury', 17: 'misc'}

# Recode in terms of integer
for num, cat_range in enumerate(icd9_ranges):
    df_diagnoses['code'] = np.where(df_diagnoses['code'].between(cat_range[0],cat_range[1]),num,df_diagnoses['code'])
    
# Convert integer to category name using diagnoses_dict
df_diagnoses['code'] = df_diagnoses['code']
df_diagnoses['category'] = df_diagnoses['code'].replace(diagnoses_dict)


# In[82]:


df_diagnoses.head()


# In[83]:


# Diagnoses for each admission
ICD_list1 = df_diagnoses.groupby('HADM_ID')['category'].apply(list).reset_index()
ICD_list1.head()


# In[84]:


ICD_list2 = pd.get_dummies(ICD_list1['category'].apply(pd.Series).stack()).max(level=0)
ICD_list2.head()


# In[85]:


df_diagnoses_new = ICD_list2.join(ICD_list1['HADM_ID'], how="outer")
df_diagnoses_new.head()


# ## Merge all preprocessed tables

# In[86]:


df_admin_new.head()


# In[87]:


df_cpt_new.head()


# In[88]:


df_icustays_new.head()


# In[89]:


df_services_new.head()


# In[90]:


df_patients_new.head()


# In[91]:


df_procedures_new.head()


# In[92]:


df_diagnoses_new.head()


# In[93]:


df = df_admin_new.merge(df_cpt_new, how='inner', on='HADM_ID')
df = df.merge(df_icustays_new, how='inner', on='HADM_ID')
df = df.merge(df_services_new, how='inner', on='HADM_ID')
df = df.merge(df_procedures_new, how='inner', on='HADM_ID')
df = df.merge(df_diagnoses_new, how='inner', on='HADM_ID')
df = df.merge(df_patients_new, how='inner', on='SUBJECT_ID')
df.head(10)


# ## Age

# In[94]:


df_age_min = df_admin[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
df_age_min.head()


# In[95]:


df_new = df_patients.merge(df_age_min, how='outer', on='SUBJECT_ID')


# In[96]:


# https://mimic.physionet.org/tutorials/intro-to-mimic-iii/
# all ages > 89 in the MIMIC database were replaced with 300

df_new['AGE'] = (df_new['ADMIT_MIN'] - df_new['DOB']).dt.days // 365
df_new['AGE'] = np.where(df_new['AGE'] <0 ,90,df_new['AGE'])
df_new['AGE'] = np.where(df_new['AGE'] == -0, 0, df_new['AGE'])
new = df_new[['SUBJECT_ID','AGE']]
new.head()


# In[97]:


df = df.merge(new, how='inner', on='SUBJECT_ID')
df.head()


# In[98]:


plt.hist(df['AGE'], bins=30, color='#2388a6', edgecolor ='black')
plt.title('Distribution of Age')
plt.ylabel('Count')
plt.xlabel('Age (years)')
plt.show()


# In[99]:


df['AGE'].isnull().sum()


# ## Preparing the Dependent variable

# ### Median LOS (days)

# In[100]:


df['LOS'].median()


# ### Removing unwanted columns

# In[101]:


df.info(verbose=True)


# In[102]:


df_final = df.copy()


# In[103]:


df_final.drop(columns=['SUBJECT_ID','HADM_ID','ADMITTIME','DOB'], inplace=True)
df_final.head()


# In[104]:


df_final['Class_LOS'] = np.where(df_final['LOS']<=df_final['LOS'].median(),0,1)
df_final.drop(columns=['LOS'], inplace=True)
df_final.head()


# ## Modeling with Median LOS

# In[105]:


cols=list(df_final.columns)


# In[106]:


cols.remove('Class_LOS')


# In[107]:


sz=df_final.index.size


# In[108]:


from sklearn.utils import shuffle


# In[109]:


df_final=shuffle(df_final)


# In[110]:


# split to training and testing
tr=df_final[:int(sz*0.8)] 
ts=df_final[int(sz*0.8):]


# ### Random Forest with 400 trees

# In[111]:


from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# In[121]:


rf=RandomForestClassifier(n_estimators =400)
rf.fit(tr[cols],tr['Class_LOS'])
probs_rf=rf.predict_proba(ts[cols])
fpr_rf, tpr_rf, thresholds_rf = roc_curve(ts['Class_LOS'],probs_rf[:,1])
auc_rf=auc(fpr_rf,tpr_rf)
plt.title('Random Forest (400 trees)')
plt.plot(fpr_rf, tpr_rf,label = "400 trees" +" (AUC= "+str(round((auc_rf),3))+")")
plt.legend(loc='center right', bbox_to_anchor=(1,0.5,0.5,0.5))


# In[122]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
y_pred = rf.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Random Forest with n trees

# In[114]:


rf = dict()
probs_rf = dict()
fpr_rf = dict()
tpr_rf = dict() 
thresholds_rf = dict()
auc_rf = dict()


# In[115]:


for i in range(10, 201, 10):
    rf[i]=RandomForestClassifier(n_estimators =i)
    rf[i].fit(tr[cols],tr['Class_LOS'])
    probs_rf[i]=rf[i].predict_proba(ts[cols])
    fpr_rf[i], tpr_rf[i], thresholds_rf[i] = roc_curve(ts['Class_LOS'],probs_rf[i][:,1])
    auc_rf[i]=auc(fpr_rf[i],tpr_rf[i])
    plt.title('Random Forest')
    plt.plot(fpr_rf[i], tpr_rf[i],label = "{} trees".format(i)+" (AUC= "+str(round((auc_rf[i]),3))+")")
    plt.legend(loc='center right', bbox_to_anchor=(1,0.5,0.5,0.5))


# In[116]:


i_rf = []
auc_rf_plot = []
for i in range(10, 201, 10):
    i_rf.append(i)
    auc_rf_plot.append(auc_rf[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Number of trees')
plt.plot(i_rf,auc_rf_plot)


# In[123]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
rf=RandomForestClassifier(n_estimators =140)
rf.fit(tr[cols],tr['Class_LOS'])
y_pred = rf.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Feature Selection using mic and chi2

# In[118]:


from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
mic = mutual_info_classif(tr[cols],tr['Class_LOS'])
mic1=chi2(tr[cols],tr['Class_LOS'])

s=pd.DataFrame()
s['att']=cols
s['mic']=mic
s['chi']=mic1[0]
s.head()


# In[115]:


rf200=RandomForestClassifier(n_estimators=400)
auc_rf200_plot_mic = dict()
for i in range(10, 101, 10):
    cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:i]
    rf200.fit(tr[cols_sel_mic],tr['Class_LOS'])
    probs_rf200=rf200.predict_proba(ts[cols_sel_mic])
    fpr_rf200, tpr_rf200, thresholds_rf200 = roc_curve(ts['Class_LOS'],probs_rf200[:,1])
    auc_rf200=auc(fpr_rf200,tpr_rf200)
    auc_rf200_plot_mic[i]=auc_rf200
    plt.title('Random Forest')
    plt.plot(fpr_rf200, tpr_rf200,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_rf200),3))+")")
    plt.legend(loc='best')


# In[116]:


i_rf200 =[]
auc_rf200_plot2 = []
for i in range(10,101,10):
    i_rf200.append(i)
    auc_rf200_plot2.append(auc_rf200_plot_mic[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using mic')
plt.plot(i_rf200,auc_rf200_plot2)


# In[151]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
rf=RandomForestClassifier(n_estimators =400)
cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:90]
rf.fit(tr[cols_sel_mic],tr['Class_LOS'])
y_pred = rf.predict(ts[cols_sel_mic])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# In[118]:


rf200=RandomForestClassifier(n_estimators=400)
auc_rf200_plot_chi = dict()
for i in range(10,101,10):
    cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:i]
    rf200.fit(tr[cols_sel_chi],tr['Class_LOS'])
    probs_rf200=rf200.predict_proba(ts[cols_sel_chi])
    fpr_rf200, tpr_rf200, thresholds_rf200 = roc_curve(ts['Class_LOS'],probs_rf200[:,1])
    auc_rf200=auc(fpr_rf200,tpr_rf200)
    auc_rf200_plot_chi[i]=auc_rf200
    plt.title('Random Forest')
    plt.plot(fpr_rf200, tpr_rf200,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_rf200),3))+")")
    plt.legend(loc='best')


# In[119]:


i_rf200 =[]
auc_rf200_plot2 = []
for i in range(10,101,10):
    i_rf200.append(i)
    auc_rf200_plot2.append(auc_rf200_plot_chi[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using chi')
plt.plot(i_rf200,auc_rf200_plot2)


# In[120]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
rf=RandomForestClassifier(n_estimators =400)
cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:60]
rf.fit(tr[cols_sel_chi],tr['Class_LOS'])
y_pred = rf.predict(ts[cols_sel_chi])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Percentage of data

# In[121]:


rf200=RandomForestClassifier(n_estimators=400)
sz=tr.index.size
auc_rf200_percent = dict()
for i in range (10, 101, 10):
    tt=tr[:int(sz*i/100.0)]
    rf200.fit(tt[cols],tt['Class_LOS'])
    probs_rf200=rf200.predict_proba(ts[cols])
    fpr_rf200, tpr_rf200, thresholds_rf200 = roc_curve(ts['Class_LOS'],probs_rf200[:,1])
    auc_rf200=auc(fpr_rf200,tpr_rf200)
    auc_rf200_percent[i] =auc_rf200
    plt.title('Random Forest')
    plt.plot(fpr_rf200, tpr_rf200, label = " {} % of train data".format(i)+" (AUC= "+str(round((auc_rf200),3))+")")
    plt.legend(loc='best')


# In[122]:


i_rf200 =[]
auc_rf200_plot_per = []
for i in range(10, 101, 10):
    i_rf200.append(i)
    auc_rf200_plot_per.append(auc_rf200_percent[i])
plt.title('Learning curve')
plt.ylabel('AUC')
plt.xlabel('Size of Data(%)')
plt.plot(i_rf200,auc_rf200_plot_per)


# In[123]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
rf=RandomForestClassifier(n_estimators =400)
tt=tr[:int(sz*90/100.0)]
rf.fit(tt[cols],tt['Class_LOS'])
y_pred = rf.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Logistic Regression 

# In[124]:


from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# In[146]:


lr=LogisticRegression()
lr.fit(tr[cols],tr['Class_LOS'])
probs_lr=lr.predict_proba(ts[cols])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(ts['Class_LOS'],probs_lr[:,1])
auc_lr=auc(fpr_lr,tpr_lr)
plt.title('Logistic Regression')
plt.plot(fpr_lr, tpr_lr,label = "AUC= "+str(round((auc_lr),3)))
plt.legend(loc='center right', bbox_to_anchor=(1,0.5,0.5,0.5))


# In[147]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
y_pred = lr.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Feature Selection using mic and chi2

# In[128]:


lr=LogisticRegression()
auc_lr_plot_mic = dict()
for i in range(10,101,10):
    cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:i]
    lr.fit(tr[cols_sel_mic],tr['Class_LOS'])
    probs_lr=lr.predict_proba(ts[cols_sel_mic])
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(ts['Class_LOS'],probs_lr[:,1])
    auc_lr=auc(fpr_lr,tpr_lr)
    auc_lr_plot_mic[i]=auc_lr
    plt.title('Logistic Regression')
    plt.plot(fpr_lr, tpr_lr,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_lr),3))+")")
    plt.legend(loc='best')


# In[129]:


i_lr =[]
auc_lr_plot2 = []
for i in range(10,101,10):
    i_lr.append(i)
    auc_lr_plot2.append(auc_lr_plot_mic[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using mic')
plt.plot(i_lr,auc_lr_plot2)


# In[130]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:100]
lr.fit(tr[cols_sel_mic],tr['Class_LOS'])
y_pred = lr.predict(ts[cols_sel_mic])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# In[131]:


lr=LogisticRegression()
auc_lr_plot_chi = dict()
for i in range(10,101,10):
    cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:i]
    lr.fit(tr[cols_sel_chi],tr['Class_LOS'])
    probs_lr=lr.predict_proba(ts[cols_sel_chi])
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(ts['Class_LOS'],probs_lr[:,1])
    auc_lr=auc(fpr_lr,tpr_lr)
    auc_lr_plot_chi[i]=auc_lr
    plt.title('Logistic Regression')
    plt.plot(fpr_lr, tpr_lr,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_lr),3))+")")
    plt.legend(loc='best')


# In[132]:


i_lr =[]
auc_lr_plot2 = []
for i in range(10,101,10):
    i_lr.append(i)
    auc_lr_plot2.append(auc_lr_plot_chi[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using chi')
plt.plot(i_lr,auc_lr_plot2)


# In[133]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:70]
lr.fit(tr[cols_sel_chi],tr['Class_LOS'])
y_pred = lr.predict(ts[cols_sel_chi])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Percentage of data

# In[134]:


lr = LogisticRegression()
sz=tr.index.size
auc_lr_percent = dict()
for i in range (10, 101, 10):
    tt=tr[:int(sz*i/100.0)]
    lr.fit(tt[cols],tt['Class_LOS'])
    probs_lr=lr.predict_proba(ts[cols])
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(ts['Class_LOS'],probs_lr[:,1])
    auc_lr=auc(fpr_lr,tpr_lr)
    auc_lr_percent[i] =auc_lr
    plt.title('Logistic Regression')
    plt.plot(fpr_lr, tpr_lr, label = " {} % of train data".format(i)+" (AUC= "+str(round((auc_lr),3))+")")
    plt.legend(loc='best')


# In[135]:


i_lr =[]
auc_lr_plot_per = []
for i in range(10, 101, 10):
    i_lr.append(i)
    auc_lr_plot_per.append(auc_lr_percent[i])
plt.title('Learning curve')
plt.ylabel('AUC')
plt.xlabel('Size of Data(%)')
plt.plot(i_lr,auc_lr_plot_per)


# In[136]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
lr = LogisticRegression()
tt=tr[:int(sz*80/100.0)]
lr.fit(tt[cols],tt['Class_LOS'])
y_pred = lr.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Naive Bayes

# In[125]:


from sklearn.naive_bayes import GaussianNB


# In[138]:


nb=GaussianNB()
nb.fit(tr[cols],tr['Class_LOS'])
probs_nb =nb.predict_proba(ts[cols])
fpr_nb, tpr_nb, thresholds_nb = roc_curve(ts['Class_LOS'],probs_nb[:,1])
auc_nb =auc(fpr_nb,tpr_nb)
plt.title('Logistic Regression')
plt.plot(fpr_nb, tpr_nb,label = "AUC= "+str(round((auc_nb),3)))
plt.legend(loc='center right', bbox_to_anchor=(1,0.5,0.5,0.5))


# In[139]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
y_pred = nb.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Feature Selection using mic and chi2

# In[140]:


nb=GaussianNB()
auc_nb_plot_mic = dict()
for i in range(10,101,10):
    cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:i]
    nb.fit(tr[cols_sel_mic],tr['Class_LOS'])
    probs_nb=nb.predict_proba(ts[cols_sel_mic])
    fpr_nb, tpr_nb, thresholds_nb = roc_curve(ts['Class_LOS'],probs_nb[:,1])
    auc_nb=auc(fpr_nb,tpr_nb)
    auc_nb_plot_mic[i]=auc_nb
    plt.title('Naive Bayes')
    plt.plot(fpr_nb, tpr_nb,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_nb),3))+")")
    plt.legend(loc='best')


# In[141]:


i_nb =[]
auc_nb_plot2 = []
for i in range(10,101,10):
    i_nb.append(i)
    auc_nb_plot2.append(auc_nb_plot_mic[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using mic')
plt.plot(i_nb,auc_nb_plot2)


# In[142]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:10]
nb.fit(tr[cols_sel_mic],tr['Class_LOS'])
y_pred = nb.predict(ts[cols_sel_mic])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# In[143]:


nb=GaussianNB()
auc_nb_plot_chi = dict()
for i in range(10,101,10):
    cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:i]
    nb.fit(tr[cols_sel_chi],tr['Class_LOS'])
    probs_nb=nb.predict_proba(ts[cols_sel_chi])
    fpr_nb, tpr_nb, thresholds_nb = roc_curve(ts['Class_LOS'],probs_nb[:,1])
    auc_nb=auc(fpr_nb,tpr_nb)
    auc_nb_plot_chi[i]=auc_nb
    plt.title('Naive Bayes')
    plt.plot(fpr_nb, tpr_nb,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_nb),3))+")")
    plt.legend(loc='best')


# In[144]:


i_nb =[]
auc_nb_plot2 = []
for i in range(10,101,10):
    i_nb.append(i)
    auc_nb_plot2.append(auc_nb_plot_chi[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using chi')
plt.plot(i_nb,auc_nb_plot2)


# In[145]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:10]
nb.fit(tr[cols_sel_chi],tr['Class_LOS'])
y_pred = nb.predict(ts[cols_sel_chi])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Percentage of data

# In[148]:


nb=GaussianNB()
sz=tr.index.size
auc_nb_percent = dict()
for i in range (10, 101, 10):
    tt=tr[:int(sz*i/100.0)]
    nb.fit(tt[cols],tt['Class_LOS'])
    probs_nb = nb.predict_proba(ts[cols])
    fpr_nb, tpr_nb, thresholds_nb = roc_curve(ts['Class_LOS'],probs_nb[:,1])
    auc_nb =auc(fpr_nb,tpr_nb)
    auc_nb_percent[i] = auc_nb
    plt.title('Naive Bayes')
    plt.plot(fpr_nb, tpr_nb, label = " {} % of train data".format(i)+" (AUC= "+str(round((auc_nb),3))+")")
    plt.legend(loc='best')


# In[149]:


i_nb =[]
auc_nb_plot_per = []
for i in range(10, 101, 10):
    i_nb.append(i)
    auc_nb_plot_per.append(auc_nb_percent[i])
plt.title('Learning curve')
plt.ylabel('AUC')
plt.xlabel('Size of Data(%)')
plt.plot(i_nb,auc_nb_plot_per)


# In[201]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
nb = GaussianNB()
tt=tr[:int(sz*10/100.0)]
nb.fit(tt[cols],tt['Class_LOS'])
y_pred = nb.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# ### Random Search Cross Validation

# ### Logistic Regression - Default parameters

# In[178]:


from sklearn.linear_model import LogisticRegression
from pprint import pprint

lr=LogisticRegression()
print('Parameters currently in use:\n')
pprint(lr.get_params())


# ### Random Search with Cross Validation

# In[258]:


from sklearn.model_selection import RandomizedSearchCV
import time

# Create regularization hyperparameter distribution using uniform distribution
C = [1.0,1.5,2.0,2.5]

# Create regularization maximum iteration space
max_iter = [100,110,120,130,140]

lr_param_grid ={'C': C,
                'max_iter':max_iter}

pprint(lr_param_grid)


# In[259]:


# Use the random grid to search for best hyperparameters

# Creating the base model to tune
lr = LogisticRegression(penalty = 'l1')

# Random search of parameters, cross validation,search across 100 different combinations, and use all available cores
lr_random = RandomizedSearchCV(estimator=lr, param_distributions=lr_param_grid, cv = 5,n_jobs=-1)

start_time = time.time()
# Fit the random search model
lr_random.fit(tr[cols], tr['Class_LOS'])

# Summarize results
print("Best training score: %f using %s" % (lr_random.best_score_, lr_random.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# ### Model evaluation function

# In[126]:


def model_eval(model, test_features, test_labels):
    y_pred = model.predict(test_features)
    print("Accuracy score : {}".format(accuracy_score(test_labels, y_pred)))
    print("Classification Report : \n {}".format(classification_report(test_labels, y_pred)))
    print("Confusion Matrix : \n {}".format(confusion_matrix(test_labels, y_pred)))
    return;


# ### Evaluate Default Model

# In[131]:


lrD = LogisticRegression()
lrD.fit(tr[cols],tr['Class_LOS'])
model_eval(lrD, ts[cols],ts['Class_LOS'])


# ### Evaluate the Best Random Search Model

# In[264]:


lrBR = LogisticRegression(penalty = 'l1',max_iter = 140, C =1.0)
lrBR.fit(tr[cols],tr['Class_LOS'])
model_eval(lrBR, ts[cols],ts['Class_LOS'])


# ### Grid Search with Cross Validation

# In[265]:


from sklearn.model_selection import GridSearchCV
import time

# Create regularization hyperparameter distribution using uniform distribution
C = [1.0,1.5]

# Create regularization maximum iteration space
max_iter = [100,110,120,130,140]

lr_param_grid ={'C': C,
                'max_iter':max_iter}

pprint(lr_param_grid)


# In[266]:


# Use the random grid to search for best hyperparameters

# Creating the base model to tune
lr = LogisticRegression(penalty ='l1')

# Random search of parameters, cross validation,search across 100 different combinations, and use all available cores
lr_random = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv = 5,n_jobs=-1)

start_time = time.time()
# Fit the random search model
lr_random.fit(tr[cols], tr['Class_LOS'])

# Summarize results
print("Best training score: %f using %s" % (lr_random.best_score_, lr_random.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# ### Evaluate the Best Grid search model

# In[136]:


lrBG = LogisticRegression(penalty = 'l1',max_iter = 100, C =1.0)
lrBG.fit(tr[cols],tr['Class_LOS'])
model_eval(lrBG, ts[cols],ts['Class_LOS'])


# ### Random Forest - Default parameters

# In[139]:


from sklearn.ensemble import RandomForestClassifier
from pprint import pprint

rf=RandomForestClassifier()
print('Parameters currently in use:\n')
pprint(rf.get_params())


# ### Random Search with Cross Validation

# In[140]:


from sklearn.model_selection import RandomizedSearchCV
import time

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 400, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


rf_param_grid ={'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(rf_param_grid)


# In[141]:


# Use the random grid to search for best hyperparameters

# Creating the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, cross validation,search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, cv = 5,n_jobs=-1)

start_time = time.time()
# Fit the random search model
rf_random.fit(tr[cols], tr['Class_LOS'])

# Summarize results
print("Best training score: %f using %s" % (rf_random.best_score_, rf_random.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# ### Evaluate Default Model

# In[142]:


rfD = RandomForestClassifier()
rfD.fit(tr[cols],tr['Class_LOS'])
model_eval(rfD, ts[cols],ts['Class_LOS'])


# ### Evaluate Best Random Search Model

# In[145]:


rfBR = RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, 
                              max_features = 'sqrt', max_depth = None, bootstrap = False)
rfBR.fit(tr[cols],tr['Class_LOS'])
model_eval(rfBR, ts[cols],ts['Class_LOS'])


# ### Grid Search with Cross Validation

# In[279]:


from sklearn.model_selection import GridSearchCV
import time

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 400, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


rf_param_grid ={'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(rf_param_grid)


# In[ ]:


# This code is supposed to generate the GridSearchCV best parameters, it took 10 hours of execution and stopped running.
""""# Use the grid to search for best hyperparameters

# Creating the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, cross validation,search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv = 5,n_jobs=-1)

start_time = time.time()
# Fit the random search model
rf_random.fit(tr[cols], tr['Class_LOS'])

# Summarize results
print("Best training score: %f using %s" % (rf_random.best_score_, rf_random.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')""""


# ### Evaluate Best Grid Search Model

# In[ ]:


""""rfGR = RandomForestClassifier(rf_random.best_params_)
rfGR.fit(tr[cols],tr['Class_LOS'])
model_eval(rfGR, ts[cols],ts['Class_LOS'])""""


# ## Refining models (Random Forest)

# In[152]:


rf200=RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, 
                              max_features = 'sqrt', max_depth = None, bootstrap = False)
auc_rf200_plot_mic = dict()
for i in range(10, 101, 10):
    cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:i]
    rf200.fit(tr[cols_sel_mic],tr['Class_LOS'])
    probs_rf200=rf200.predict_proba(ts[cols_sel_mic])
    fpr_rf200, tpr_rf200, thresholds_rf200 = roc_curve(ts['Class_LOS'],probs_rf200[:,1])
    auc_rf200=auc(fpr_rf200,tpr_rf200)
    auc_rf200_plot_mic[i]=auc_rf200
    plt.title('Random Forest')
    plt.plot(fpr_rf200, tpr_rf200,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_rf200),3))+")")
    plt.legend(loc='best')


# In[153]:


i_rf200 =[]
auc_rf200_plot2 = []
for i in range(10,101,10):
    i_rf200.append(i)
    auc_rf200_plot2.append(auc_rf200_plot_mic[i])
plt.title('Performance curve')
plt.ylabel('AUC')
plt.xlabel('Top attributes using mic')
plt.plot(i_rf200,auc_rf200_plot2)


# In[155]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
rf=RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, 
                              max_features = 'sqrt', max_depth = None, bootstrap = False)
cols_sel_mic = s.sort_values('mic', ascending=False)['att'][:100]
rf.fit(tr[cols_sel_mic],tr['Class_LOS'])
y_pred = rf.predict(ts[cols_sel_mic])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# In[156]:


rf200=RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, 
                              max_features = 'sqrt', max_depth = None, bootstrap = False)
auc_rf200_plot_chi = dict()
for i in range(10,101,10):
    cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:i]
    rf200.fit(tr[cols_sel_chi],tr['Class_LOS'])
    probs_rf200=rf200.predict_proba(ts[cols_sel_chi])
    fpr_rf200, tpr_rf200, thresholds_rf200 = roc_curve(ts['Class_LOS'],probs_rf200[:,1])
    auc_rf200=auc(fpr_rf200,tpr_rf200)
    auc_rf200_plot_chi[i]=auc_rf200
    plt.title('Random Forest')
    plt.plot(fpr_rf200, tpr_rf200,label ="Top"+" {} attributes".format(i)+" (AUC= "+str(round((auc_rf200),3))+")")
    plt.legend(loc='best')


# In[160]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
rf=RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, 
                              max_features = 'sqrt', max_depth = None, bootstrap = False)
cols_sel_chi = s.sort_values('chi', ascending=False)['att'][:100]
rf.fit(tr[cols_sel_chi],tr['Class_LOS'])
y_pred = rf.predict(ts[cols_sel_chi])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# In[161]:


rf200=RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, 
                              max_features = 'sqrt', max_depth = None, bootstrap = False)
sz=tr.index.size
auc_rf200_percent = dict()
for i in range (10, 101, 10):
    tt=tr[:int(sz*i/100.0)]
    rf200.fit(tt[cols],tt['Class_LOS'])
    probs_rf200=rf200.predict_proba(ts[cols])
    fpr_rf200, tpr_rf200, thresholds_rf200 = roc_curve(ts['Class_LOS'],probs_rf200[:,1])
    auc_rf200=auc(fpr_rf200,tpr_rf200)
    auc_rf200_percent[i] =auc_rf200
    plt.title('Random Forest')
    plt.plot(fpr_rf200, tpr_rf200, label = " {} % of train data".format(i)+" (AUC= "+str(round((auc_rf200),3))+")")
    plt.legend(loc='best')


# In[162]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
rf=RandomForestClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, 
                              max_features = 'sqrt', max_depth = None, bootstrap = False)
tt=tr[:int(sz*70/100.0)]
rf.fit(tt[cols],tt['Class_LOS'])
y_pred = rf.predict(ts[cols])
print("Accuracy score : {}".format(accuracy_score(ts['Class_LOS'], y_pred)))
print("Classification Report : \n {}".format(classification_report(ts['Class_LOS'], y_pred)))
print("Confusion Matrix : \n {}".format(confusion_matrix(ts['Class_LOS'], y_pred)))


# In[ ]:




