import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


#! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz


# In[4]:


# To better display 3 samples, I set the max column width to the max of review string.
# This process can show the whole review in the table
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 20)


# ## Read Data

# In[5]:


# read the data to a pandas dataframe
df = pd.read_table('amazon_reviews_us_Kitchen_v1_00.tsv', error_bad_lines=False,warn_bad_lines=False)


# ## Keep Reviews and Ratings

# In[6]:


# I select the first 3 reviews as samples to show
from IPython.display import display
raw_data = df[['star_rating','review_body']]
show_data = raw_data.head(3)
print("There are 3 sample reviews:")
display(show_data)
print("For the sample that I selected, all of them received 5 star ratings.")


# # Labelling Reviews:
# ## The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'

# In[7]:


# Label reviews with rating 4,5 to class 1 and reviews with rating 1,2 to class 0
df1 = raw_data[raw_data['star_rating'] > 3].dropna()
df2 = raw_data[raw_data['star_rating'] < 3].dropna()
df1['label'] = 1
df2['label'] = 0

# count the number of reviews for each star rating and compute statistics
count1 = raw_data[raw_data['star_rating'] == 1].iloc[:,0].size
count2 = raw_data[raw_data['star_rating'] == 2].iloc[:,0].size
count3 = raw_data[raw_data['star_rating'] == 3].iloc[:,0].size
count4 = raw_data[raw_data['star_rating'] == 4].iloc[:,0].size
count5 = raw_data[raw_data['star_rating'] == 5].iloc[:,0].size
answer_str = 'The number of reviews for class 0, 1 and neutral reviews (rating 3) are '
answer_str = answer_str+str(count1+count2)+', '+str(count4+count5)+', and '+str(count3)+'. '

avg0 = (count1+(count2*2))/(count1+count2)
avg1 = ((count4*4)+(count5*5))/(count4+count5)
answer_str = answer_str+'The average rating for class 0, 1 and neutral reviews (rating 3) are '
answer_str = answer_str+str(avg0)+', 3, and '+str(avg1)+'. '
print(answer_str)


#  ## We select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews.
#
#

# In[8]:


# Select 100,000 positive and 100,000 negative reviews.
# In order to get same result everytime, the random_state is set to be a constant.
# concate positive and negative review dataframe at the end.
pos_df = df1.sample(n = 100000, random_state=2)
neg_df = df2.sample(n = 100000, random_state=2)
all_data = pd.concat([pos_df, neg_df])


# # Data Cleaning
#
# ## Convert the all reviews into the lower case.

# In[9]:


# Before cleaning data, keep a copy of it for future computation
before_clean_data = all_data.copy()

# convert all reviews to lower case
all_data['review_body'] = all_data['review_body'].str.lower()


# ## remove the HTML and URLs from the reviews

# In[10]:


# Remove HTML
all_data['review_body'] = all_data['review_body'].apply(lambda text: BeautifulSoup(text).get_text())


# In[11]:


# Remove URL by remove all word start with 'http:' or 'https:', then remove all word start with 'www.' and end with '.com'
all_data['review_body'] = all_data['review_body'].apply(lambda text: re.sub(r'https?:\S+', '', text))
all_data['review_body'] = all_data['review_body'].apply(lambda text: re.sub(r'www.\S+.com', '', text))


# ## perform contractions on the reviews.

# In[12]:


# I manually code the contraction function by replace specific expression with their expand version.
def contractionfunction(s):
    # specific
    s = re.sub(r"won\'t", "will not", s)
    s = re.sub(r"can\'t", "can not", s)
    s = re.sub(r'ain\'t', 'are not', s)

    # general
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r'(\w+)\'re', '\g<1> are', s)
    s = re.sub(r'(\w+)\'s', '\g<1> is', s)
    s = re.sub(r'(\w+)\'d', '\g<1> would', s)
    s = re.sub(r'(\w+)\'ll', '\g<1> will', s)
    s = re.sub(r'(\w+)\'t', '\g<1> not', s)
    s = re.sub(r'(\w+)\'ve', '\g<1> have', s)
    s = re.sub(r'(\w+)\'m', '\g<1> am', s)
    return s

all_data['review_body'] = all_data['review_body'].apply(contractionfunction)


# ## remove non-alphabetical characters

# In[13]:


# remove all non-alphabetical characters
regex = re.compile('[^a-zA-Z]')
all_data['review_body'] = all_data['review_body'].apply(lambda text: regex.sub(' ', text))


# ## Remove the extra spaces between the words

# In[14]:


# Remove all extra spaces
all_data['review_body'] = all_data['review_body'].apply(lambda text: re.sub(' +', ' ', text))


# In[15]:


# compute values and report
before_clean = sum(before_clean_data["review_body"].str.len())/(before_clean_data.iloc[:,0].size)
after_clean = sum(all_data["review_body"].str.len())/(all_data.iloc[:,0].size)
print('Before data cleaning, the average length of review is '+str(before_clean)+'. After data cleaning, the average length of review is '+str(after_clean)+'. ')


# # Pre-processing

# ## remove the stop words

# In[17]:


# Before pre-processing, keep a copy of old data for future computation
before_clean_data1 = all_data.copy()

# create a stop word list for english
from nltk.corpus import stopwords
words_list = stopwords.words('english')

# split each review into words and check them one by one, remove the word if it is a stop word,
# and concate the words finally
def remove_stop(s):
    pieces = s.split()
    result = ''
    for each_word in pieces:
        if each_word not in words_list:
            result = result+' '+each_word
    if len(result)>0:
        result = result[1:]
    return result

all_data['review_body'] = all_data['review_body'].apply(remove_stop)


# ## perform lemmatization

# In[18]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# split each review into words and perform lemmatization one by one, and concate the words finally
def lemmatization(s):
    pieces = s.split()
    result = ''
    for each_word in pieces:
        temp = lemmatizer.lemmatize(each_word)
        result = result+' '+temp
    if len(result)>0:
        result = result[1:]
    return result

all_data['review_body'] = all_data['review_body'].apply(lemmatization)


# In[19]:


# compute values and report
show_data_before = before_clean_data.head(3)
print("There are 3 sample reviews before data cleaning and preprocessing:")
display(show_data_before)
show_data_after = all_data.head(3)
print("There are 3 sample reviews after data cleaning and preprocessing:")
display(show_data_after)

before_clean = sum(before_clean_data1["review_body"].str.len())/(before_clean_data1.iloc[:,0].size)
after_clean = sum(all_data["review_body"].str.len())/(all_data.iloc[:,0].size)
print('Before preprocessing, the average length of review is '+str(before_clean)+'. After preprocessing, the average length of review is '+str(after_clean)+'. ')


# # TF-IDF Feature Extraction

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
review_list = all_data['review_body'].tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(review_list)
#print(X.shape)
vector_df = pd.DataFrame.sparse.from_spmatrix(X)


# # Perceptron

# In[21]:


# split train and test, to keep classes distribute evenly, I set the stratify to label list.
from sklearn.model_selection import train_test_split
all_label = all_data['label']
x_train, x_test, y_train, y_test = train_test_split(vector_df,all_label, test_size=0.2,random_state=2,stratify=all_label)


# In[22]:


from sklearn.linear_model import Perceptron
clf = Perceptron(random_state=2)
clf.fit(x_train, y_train)


# In[23]:


# train + test prediction

# train
y_pred = clf.predict(x_train)
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = '\nFor Perceptron model, the accuracy, precision, recall and f1-score of training dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+". "

# test
y_pred = clf.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = answer_str+'The accuracy, precision, recall and f1-score of testing dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+"."

print(answer_str)


# # SVM

# In[24]:


from sklearn.svm import LinearSVC
lsvc = LinearSVC(random_state=2)
lsvc.fit(x_train, y_train)


# In[25]:


# train + test prediction

# train
y_pred = lsvc.predict(x_train)
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = '\nFor SVM model, the accuracy, precision, recall and f1-score of training dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+". "

# test
y_pred = lsvc.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = answer_str+'The accuracy, precision, recall and f1-score of testing dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+"."

print(answer_str)


# # Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(x_train, y_train)


# In[27]:


# train + test prediction

# train
y_pred = clf.predict(x_train)
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = '\nFor Logistic Regression model, the accuracy, precision, recall and f1-score of training dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+". "

# test
y_pred = clf.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = answer_str+'The accuracy, precision, recall and f1-score of testing dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+"."

print(answer_str)


# # Naive Bayes

# In[28]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_train, y_train)


# In[29]:


# train + test prediction

# train
y_pred = clf.predict(x_train)
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = '\nFor Multinomial Naive Bayes model, the accuracy, precision, recall and f1-score of training dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+". "

# test
y_pred = clf.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tn+tp)/(tn+fp+fn+tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*((precision*recall)/(precision+recall))

answer_str = answer_str+'The accuracy, precision, recall and f1-score of testing dataset are '
answer_str = answer_str+str(accuracy)+", "+str(precision)+", "+str(recall)+", "+str(f1_score)+"."

print(answer_str)
