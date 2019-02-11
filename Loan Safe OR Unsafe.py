
# coding: utf-8

# In[1]:

import graphlab
graphlab.canvas.set_target('ipynb')


# In[2]:

loans = graphlab.SFrame('lending-club-data.gl/')


# In[3]:

loans.column_names()


# In[4]:

loans['grade'].show()


# In[5]:

loans['home_ownership'].show()


# In[6]:

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')


# In[7]:

loans['safe_loans'].show(view = 'Categorical')


# In[9]:

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # 90 day or worse rating
            'revol_util',                # % of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]


# In[10]:

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)


# In[11]:

print "% of safe loans  :", (len(safe_loans_raw)/float(len(safe_loans_raw) + len(risky_loans_raw)))
print "% of risky loans :", (len(risky_loans_raw)/float(len(safe_loans_raw) + len(risky_loans_raw)))


# In[12]:

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)


# In[13]:

print "% of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "% of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)


# In[14]:

train_data, validation_data = loans_data.random_split(.8, seed=1)


# In[15]:

decision_tree_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                target = target, features = features)


# In[16]:

decision_tree_model.show(view="Tree")


# In[17]:

small_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 2)


# In[18]:


small_model.show(view="Tree")


# In[19]:

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data


# In[20]:

decision_tree_model.predict(sample_validation_data)


# In[21]:

(sample_validation_data['safe_loans'] == decision_tree_model.predict(sample_validation_data)).sum()/float(len(sample_validation_data))


# In[22]:

decision_tree_model.predict(sample_validation_data, output_type='probability')


# In[23]:

small_model.predict(sample_validation_data, output_type='probability')


# In[24]:

sample_validation_data[1]


# In[25]:

small_model.show(view="Tree")


# In[26]:

small_model.predict(sample_validation_data[1])


# In[27]:

print small_model.evaluate(train_data)['accuracy']
print decision_tree_model.evaluate(train_data)['accuracy']


# In[28]:

print small_model.evaluate(validation_data)['accuracy']
print decision_tree_model.evaluate(validation_data)['accuracy']


# In[29]:

big_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                   target = target, features = features, max_depth = 10)


# In[30]:

print big_model.evaluate(train_data)['accuracy']
print big_model.evaluate(validation_data)['accuracy']


# In[31]:

predictions = decision_tree_model.predict(validation_data)


# In[32]:

decision_tree_model.show(view='Evaluation')


# In[33]:

len(predictions)


# In[34]:

false_positives = (validation_data[validation_data['safe_loans'] != predictions]['safe_loans'] == -1).sum()
print false_positives


# In[ ]:



