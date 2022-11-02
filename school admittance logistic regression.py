#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


raw_data = pd.read_csv('1.0_train_data_set.csv')


# In[3]:


data = raw_data.copy()


# In[4]:


data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No':0})
data['Gender'] = data['Gender'].map({'Female' :1, 'Male': 0})
data.head()


# In[5]:


y = data['Admitted']
x1 = data['SAT']


# In[6]:


x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()


# $ \log({odds}) = -0.69.9128 + 0.042 (SAT) $
# 
# $\Delta_{odds} = e^{β_k}$
#   
#  $ \log({odds_2}) = -0.69.9128 + 0.042 (SAT_2) $
#  
# $ \log({odds_1}) = -0.69.9128 + 0.042 (SAT_1) $
#   
# $ \log(\frac{odds_2}{odds_1})  = 0.042 (SAT_2 - SAT_1) $
# 
# $ \log(\frac{odds_{SAT_2}}{odds_{SAT_1}}) = 0.042 $
# 
# 
# $ odds_{SAT_2} =1.042 (odds_{SAT_1}) $
# 
# $ odds_{SAT_2} = 104.2\% (odds_{SAT_1}) $

# In[26]:


print(np.exp(0.042)) 
print(np.exp(0.42))


# In[35]:


llr_pvalue = round(5.805e-42,3)
llr_pvalue


# #### Observations:
# - The logit model according to the logit regression table is shown above. 
# - The model is significant and the SAT variable is significant too. 
# - LLR p-value is 0.00 so our model is statistically different from LL-Null. 
# - We created 2 other equations one for $ \log(odds_1) $ with $(SAT_1)$ variable and one for $\log(odds_2)$ with $(SAT_2)$ variable.
# - The logistic regression coefficient β associated with a predictor X is the expected change in log odds of having the outcome per unit change in X. So increasing the predictor by 1 unit (or going from 1 level to the next) multiplies the odds of having the outcome by $e^β$ according to the following equation $\Delta_{odds} = e^{β_k}$
# - Let $SAT_2$ equal to 1 and $SAT_1$ equal to 0. This way we can see outcome for increasing the predictor by 1 unit. 
# - If we take the exponential of both sides
# - $ odds_{SAT{2}} =1.042(odds_{SAT_{1}}) $
# - When the SAT score increase by 1, the odds of getting admittance increase by 4.2%.
# - If the SAT score increase by 10, the odds of getting addmittance increase by 52%.

# In[8]:


y = data['Admitted']
x1 = data['Gender']


# In[9]:


x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()


# In[36]:


np.exp(2.0786)


# In[37]:


llr_pvalue = round(6.283e-10,3)
llr_pvalue


# 
# 

# $ \log({odds}) = -0.64 + 2.0786 (gender) $
# 
# $\Delta_{odds} = e^{β_k}$
#   
#  $ \log({odds_2}) = -0.64 + 2.0786 (gender_2) $
#  
# $ \log({odds_1}) = -0.64 + 2.0786 (gender_1) $
#   
# $ \log(\frac{odds_2}{odds_1}) = 2.0786 (gender_2 - gender_1) $
# 
# $ \log(\frac{odds_{female}}{odds_{male}}) = 2.0786 $
# 
# 
# $ odds_{female} =7.99 (odds_{male}) $

# #### Observations:
# - The logit model according to the logit regression table is shown above. 
# - The model is significant and the gender variable is significant too.
# - LLR p-value is 0.00 so our model is statistically different from LL-Null. 
# - We created 2 other equations one for $ \log(odds_1) $ with $(gender_1)$ variable and one for $\log(odds_2)$ with $(gender_2)$ variable.
# -  $\Delta_{odds} = e^{β_k}$
# - Let $gender_2$ equal to 1 (female) and $gender_1$ equal to 0 (male).
# - If we take the exponential of both sides
# - $ odds_{female} =7.99 (odds_{male}) $
# - According to the equation the odds of a female to get admitted are 7.99 the odds of a male to get admitted.
# 
# 

# In[11]:


y = data['Admitted']
x1 = data[['SAT', 'Gender']]


# In[12]:


x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()


# In[38]:


np.exp(1.9449)


# In[40]:


llr_pvalue = round(5.118e-42,3)
llr_pvalue


# ### Observations:
# - This regression has a much higher Log-Likelihood value than the previous one, meaning current model is a better version. 
# - Gender variable is still significant but it is no longer 0.00. It is a less significant variable in this model than the previous one.
# - The new coefficient of gender is 1.9449 and it's exponential is 6.99. Which indicates that given the same SAT score the odds of a female to get admitted are 6.99 the odds of a male to get admitted.
# - $ \log(\frac{odds_{female}}{odds_{male}}) = 1.9449 $

# In[14]:


np.set_printoptions(formatter = {'float' :lambda x: "{0:0.2f}".format(x)})
results_log.predict()


# In[15]:


np.array(data['Admitted'])


# In[16]:


results_log.pred_table()


# In[17]:


cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
cm_df


# ### Confusion Matrix
# - For 69 observations the model predicted 0 and the true value was 0
# - For 90 observations the model predicted 1 and the true value was 1
# - For 4 observations the model predicted 0 while the true value was 1
# - For 5 observations the model predicted 1 and the true value was 0
# - Accurancy of this train set is 94.6%

# In[18]:


# accurancy of the model 
cm = np.array(cm_df)
accurancy_train = (cm[0,0] + cm[1,1])/cm.sum()
accurancy_train


# In[19]:


test = pd.read_csv('2.03.+Test+dataset.csv')
test.head()


# In[20]:


test['Admitted'] = test['Admitted'].map({'Yes': 1, 'No': 0})
test['Gender'] = test['Gender'].map({'Female': 1, 'Male': 0})
test


# In[21]:


test_actual = test['Admitted']
test_data = test.drop(['Admitted'],axis=1)
test_data = sm.add_constant(test_data)
test_data


# In[22]:


def confusion_matrix(data,actual_values,model):    
    
        pred_values = model.predict(data)       
        bins=np.array([0,0.5,1])       
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]   
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()       
        return cm, accuracy


# In[44]:


cm = confusion_matrix(test_data,test_actual,results_log)
print('confusion matrix :')
print(cm[0])

print('accuracy : ' +str(cm[1]))


# In[24]:


cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# In[45]:


print ('Missclassification rate: '+str((2)/19))

