
# coding: utf-8

# In[70]:

import numpy as np
import pandas as pd
from patsy import dmatrices
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics


# In[71]:

# Load dataset
related =  pd.read_csv("./data_vectorised/related.csv")
infection =  pd.read_csv("./data_vectorised/infection.csv")
self =  pd.read_csv("./data_vectorised/self.csv")


# ## Data Exploration

# In[72]:

#related.groupby('RESULT').mean()
#infection.groupby('RESULT').mean()
#self.groupby('RESULT').mean()


# ## Logistic Regression

# In[73]:

related.head()


# In[74]:

y, X = dmatrices("RESULT ~ flu + gett + swin + shot + s + nt + think + bird + sick + get", related, return_type = 'dataframe')
# flatten y into a 1-D array
y = np.ravel(y)


# In[75]:

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X,y)

# check the accuracy on the training set
model.score(X, y)


# ## Model Evaluation Using a Validation Set

# In[76]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)


# We now need to predict class labels for the test set. We will also generate the class probabilities, just to take a look.

# In[79]:

# predict class labels for the test set
predicted = model2.predict(X_test)
print predicted


# In[55]:

# generate class probabilities
probs = model2.predict_proba(X_test)
print probs


# As can be seen, the classifier is predicting a 1 any time the probability in the second column is greater than 0.5.
# 
# Now let's generate some evaluation metrics.

# In[56]:

# generate evaluation metrics
print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])


# The accuracy is 62%, which is the same as I experienced when training and predicting on the same data.
# We can also see the confusion matrix and a classification report with other metrics

# In[60]:

print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)


# ## Model Evaluation Using Cross-Validation

# Now let's try 10-fold cross-validation, to see if the accuracy holds up more rigorously.

# In[61]:

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()


# It's still performing at 62% accuracy'

# ## Predicting the Probability that a tweet is related to influenza

# In[87]:

X = np.array([1,3,0,1,0,0,0,0,0,1,1])
X.reshape(-1, 1)
model.predict_proba(X)


# In[ ]:




# In[ ]:



