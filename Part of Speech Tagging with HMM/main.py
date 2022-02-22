import pandas as pd


# ## Task1

# In[2]:


# Load training data
train = pd.read_table('train',header=None, names=['index','word','tag'])
# count the occurence of each word
train_word_count = train.groupby('word').size().reset_index(name='count')
# add occurence of each word as a new column to filter out '<unk>'
train_df = pd.merge(train, train_word_count, on='word')
#train_df.head(3)


# In[3]:


# set threshold and filter out '<unk>', the new word column is 'filterd_word'
threshold = 2
def label_unk(row):
    if row['count']<threshold:
        return '<unk>';
    else:
        return row['word'];

train_df['filterd_word'] = train_df.apply (lambda row: label_unk(row), axis=1)
#train_df.head(3)


# In[4]:


#### Task1 output ####

# count the occurence of each filtered word
task1_output = train_df.groupby('filterd_word').size().reset_index(name='occurance')
# sort dataframe based on occurence of each word
task1_output = task1_output.sort_values(by='occurance', ascending=False)

# let word '<unk>' on the first row of dataframe
task1_output = task1_output.reset_index(drop=True)
unk_index = task1_output.index[task1_output['filterd_word'] == '<unk>'].tolist()
unk_index = unk_index[0]
task1_output["new"] = range(1,len(task1_output)+1)
task1_output.loc[task1_output.index==unk_index, 'new'] = 0
task1_output = task1_output.sort_values("new").drop('new', axis=1)
task1_output = task1_output.reset_index(drop=True)

# add index to the dataframe
task1_output["index"] = range(1,len(task1_output)+1)
#task1_output.head(3)


# In[5]:


print('The threshold I selected for unknown words replacement is '+str(threshold))
print('The number of vocabulary is '+str(task1_output.iloc[:,0].size))
print('The total size of vocabulary is '+str(task1_output['occurance'].sum()))
print('The occurence of "<unk>" is '+str(task1_output.at[0,'occurance']))


# In[6]:


# output to vocab.txt
task1_output_list = task1_output.values.tolist()
with open('vocab.txt', 'w') as f:
    for each_output in task1_output_list:
        word_type = each_output[0]
        occurence = each_output[1]
        index = each_output[2]
        f.write(word_type+'\t'+str(index)+'\t'+str(occurence))
        f.write('\n')
f.close()


# In[7]:


# split '<unk>' to two categories to improve accuracy
def label_unk(row):
    if row['count']<threshold:
        # if the word start with digit, then it is '<unk_digit>' word
        if row['word'][0]>= '0' and row['word'] <= '9':
            return '<unk_digit>';
        # if the word does not start with digit, then it is regular'<unk>' word
        else:
            return '<unk>';
    else:
        return row['word'];

train_df['filterd_word'] = train_df.apply (lambda row: label_unk(row), axis=1)


# In[8]:


# keep a list of words for future computation
all_word = task1_output['filterd_word'].tolist()


# In[9]:


report_unk = train_df.groupby('filterd_word').size().reset_index(name='occurance')
report_unk = report_unk[report_unk['filterd_word']=='<unk>'].reset_index(drop=True)

print('I split "<unk>" to two categories, which is "<unk>" and "<unk_digit>" to improve accuracy, ')
print('The answers below are based on the data that after my preprocessed:')

print('The number of vocabulary is '+str(len(all_word)))
print('The total size of vocabulary is '+str(task1_output['occurance'].sum()))
print('The occurence of "<unk>" is '+str(report_unk.at[0,'occurance']))


# ## Task2

# In[10]:


# add tag2 to dataframe in order to create transition tag pair
# tag2 is the one row shift of original tag, which can represent next tag
train_tag = train.copy()
train_tag['tag2']=train_tag['tag']
train_tag['tag2'] = train_tag.tag2.shift(-1,fill_value=train_tag.at[0,'tag'])
#train_tag.head(3)


# In[11]:


# combine two tags into tuple to get representation of transition
def add_tag_pair(row):
    return (row['tag'],row['tag2']);

train_tag['tag_pair'] = train_tag.apply (lambda row: add_tag_pair(row), axis=1)
#train_tag.head(3)


# In[12]:


# count the number of occurence of each transition and save to an list for future computation
train_tagPair_count = train_tag.groupby('tag_pair').size().reset_index(name='count')
pair_list = train_tagPair_count.values.tolist()
#train_tagPair_count.head(3)


# In[13]:


# count the number of occurence of each tag and save to an dictionary for future computation
train_tag_count = train_tag.groupby('tag').size().reset_index(name='count')
tag_count_dict = train_tag_count.set_index('tag').to_dict()['count']
#train_tag_count.head(3)


# In[14]:


# compute transition dictionary
transition = {}
# for transition tag pair, get its occurence from previous dictionary
# also get start tag's occurence from previous dictionary
# calculate the transition value and add it to transition dictionary as value with tag pair as key
for each_pair in pair_list:
    tag_pair = each_pair[0]
    tag_pair_count = each_pair[1]
    first_tag = tag_pair[0]
    tag_count = tag_count_dict[first_tag]
    transition[tag_pair]=tag_pair_count/tag_count
#transition


# In[15]:


# create (tag,word) pair to compute emission
train_word_tag = train_df[['filterd_word','tag']]
def add_tag_word_pair(row):
    return (row['tag'],row['filterd_word']);

train_word_tag['tag_word_pair'] = train_word_tag.apply (lambda row: add_tag_word_pair(row), axis=1)
#train_word_tag.head(3)


# In[16]:


# count the number of occurence of each emission and save to an list for future computation
train_tag_word_pair_count = train_word_tag.groupby('tag_word_pair').size().reset_index(name='count')
pair_list = train_tag_word_pair_count.values.tolist()
#train_tag_word_pair_count.head(3)


# In[17]:


# compute emission dictionary
emission = {}
# for emission (tag,word) pair, get its occurence from previous dictionary
# also get start tag's occurence from previous dictionary
# calculate the emission value and add it to emission dictionary as value with (tag,word) pair as key
for each_pair in pair_list:
    tag_pair = each_pair[0]
    tag_pair_count = each_pair[1]
    first_tag = tag_pair[0]
    tag_count = tag_count_dict[first_tag]
    emission[tag_pair]=tag_pair_count/tag_count
#emission


# In[18]:


# output json file
import json
# convert tuple key to string in order to output to json file
keys_values = transition.items()
transition_output = {str(key): value for key, value in keys_values}
keys_values2 = emission.items()
emission_output = {str(key): value for key, value in keys_values2}

with open('hmm.json', 'w') as fp:
    json.dump(transition_output, fp, indent=2)
    json.dump(emission_output, fp, indent=2)


# In[19]:


print('The size of transition is '+str(len(transition)))
print('The size of emission is '+str(len(emission)))


# ## Task3

# In[20]:


# load dev data
dev = pd.read_table('dev',header=None, names=['index','word','correct_tag'])#,delim_whitespace=True,header=0
#dev.head(3)


# In[21]:


# convert dev data to list for future computation
dev_test = dev[['index','word']]
all_tags = list(tag_count_dict.keys())
index_list = dev_test.values.tolist()


# In[22]:


#### Greedy Algorithm ####
previous_tag="."
result_tag_list = []
# for each word, compute the argmax(transition*emission)
for each_row in index_list:
    current_word = each_row[1]
    # For some special character, the tag is fixed, so we can skip the computation step to save time
    if current_word=='?':
        predict_tag='.'
    elif current_word=='--' or current_word=='...':
        predict_tag=':'
    elif current_word==')' or current_word=='}':
        predict_tag='-RRB-'
    elif current_word=='(' or current_word=='{':
        predict_tag='-LRB-'
    elif current_word in ['.',',','$','``']:
        predict_tag=current_word
    else:
        # if the word is not in word dictionary, we treat it as unknown
        if current_word not in all_word:
            # treat the word as '<unk_digit>' if the word start with a digit
            if current_word[0]>= '0' and current_word[0] <= '9':
                current_word='<unk_digit>'
            else:
                current_word='<unk>'

        # for each tag, compute the transmission*emission value and take the argmax as predicted tag
        te_list = []
        for each_tag in all_tags:
            t_value = transition.get((previous_tag,each_tag),0)
            e_value = emission.get((each_tag,current_word),0)
            te_list.append(t_value*e_value)
        max_index = te_list.index(max(te_list))
        predict_tag = all_tags[max_index]

    # save predicted tag as input of next round computation
    previous_tag = predict_tag
    # save answer for current word
    result_tag_list.append(predict_tag)


# In[23]:


# add predicted tag as a new column to compute accuracy
dev['greedy_tag']=result_tag_list
num_correct = dev[dev['correct_tag']==dev['greedy_tag']].iloc[:,0].size
print('The accuracy of applying Greedy algorithm on dev data is '+str(num_correct/dev.iloc[:,0].size))


# In[24]:


# predicting tags for test data
# load test data
test = pd.read_table('test',header=None, names=['index','word'])
index_list_test = test.values.tolist()

# Apply Greedy Algorithm to test data
previous_tag="."
result_tag_list = []
# for each word, compute the argmax(transition*emission)
for each_row in index_list_test:
    current_word = each_row[1]
    # For some special character, the tag is fixed, so we can skip the computation step to save time
    if current_word=='?':
        predict_tag='.'
    elif current_word=='--' or current_word=='...':
        predict_tag=':'
    elif current_word==')' or current_word=='}':
        predict_tag='-RRB-'
    elif current_word=='(' or current_word=='{':
        predict_tag='-LRB-'
    elif current_word in ['.',',','$','``']:
        predict_tag=current_word
    else:
        # if the word is not in word dictionary, we treat it as unknown
        if current_word not in all_word:
            # treat the word as '<unk_digit>' if the word start with a digit
            if current_word[0]>= '0' and current_word[0] <= '9':
                current_word='<unk_digit>'
            else:
                current_word='<unk>'

        # for each tag, compute the transmission*emission value and take the argmax as predicted tag
        te_list = []
        for each_tag in all_tags:
            t_value = transition.get((previous_tag,each_tag),0)
            e_value = emission.get((each_tag,current_word),0)
            te_list.append(t_value*e_value)
        max_index = te_list.index(max(te_list))
        predict_tag = all_tags[max_index]

    # save predicted tag as input of next round computation
    previous_tag = predict_tag
    # save answer for current word
    result_tag_list.append(predict_tag)


# In[25]:


# add predicted tag as a new column and output it
test['greedy_tag']=result_tag_list

# output to greedy.out
task3_output_list = test.values.tolist()
first_flag = True
with open('greedy.out', 'w') as f:
    for each_output in task3_output_list:
        index = each_output[0]
        word_type = each_output[1]
        tag = each_output[2]
        if index==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')

        f.write(str(index)+'\t'+word_type+'\t'+tag)
        f.write('\n')
f.close()


# ## Task4

# In[26]:


# because the Viterbi algorithm predict sentence by sentence, so data is splited into list of sentences
def get_sentences(i_list):
    current_sentence = []
    result = []
    for each_pair in i_list:
        if each_pair[0]!=1:
            current_sentence.append(each_pair)
        else:
            result.append(current_sentence)
            current_sentence = [each_pair]
    result = result[1:]
    result.append(current_sentence)
    return result
all_sentences = get_sentences(index_list)
#all_sentences


# In[27]:


#### Viterbi Algorithm ####
viterbi_result_tag_list = []
num_of_tags = len(all_tags)

# for each sentence, compute tags for each word
for each_sentence in all_sentences:

    # initialize a Table to keep pai value, each cell contain (previous_tag,current_tag,max(transition*emission))
    # for the first column, the previous_tag is '.'
    pai = []
    previous_tag="."
    each_row = each_sentence[0]
    current_word = each_row[1]

    # if the word is not in word dictionary, we treat it as unknown
    if current_word not in all_word:
        # treat the word as '<unk_digit>' if the word start with a digit
        if current_word[0]>= '0' and current_word[0] <= '9':
            current_word='<unk_digit>'
        else:
            current_word='<unk>'

    # initialize first column of pai based on first word of sentence
    current_col = []
    # for each tag, compute (transition*emission), save to table pai
    for each_tag in all_tags:
        t_value = transition.get((previous_tag,each_tag),0)
        e_value = emission.get((each_tag,current_word),0)
        current_col.append((previous_tag,each_tag,t_value*e_value))
    pai.append(current_col)


    # compute other columns recursively
    for i in range(1,len(each_sentence)):
        each_row = each_sentence[i]
        current_word = each_row[1]
        # if the word is not in word dictionary, we treat it as unknown
        if current_word not in all_word:
            # treat the word as '<unk_digit>' if the word start with a digit
            if current_word[0]>= '0' and current_word[0] <= '9':
                current_word='<unk_digit>'
            else:
                current_word='<unk>'


        previous_col = pai[i-1]
        current_col = []

        # for each current tag, compute (transition*emission)for each previous tag and find argmax of it,
        # then save (previous_tag,current_tag,max(pai*transition*emission) to table pai
        for each_tag in all_tags:
            test_max_list = []
            for each_previous_pair in previous_col:
                each_previous_tag = each_previous_pair[1] # get previous tag
                each_previous_value = each_previous_pair[2]  # get previous pai value
                t_value = transition.get((each_previous_tag,each_tag),0)
                e_value = emission.get((each_tag,current_word),0)
                test_max_list.append(t_value*e_value*each_previous_value)

            # find argmax of (pai*transition*emission)
            max_value = max(test_max_list)
            max_index = test_max_list.index(max_value)
            previous_tag = all_tags[max_index]
            current_col.append((previous_tag,each_tag,max_value))
        pai.append(current_col)



    # traverse back
    current_sentence_tag = []

    # start from last column, find max pai value and its corresponding tag
    current_col = pai[len(each_sentence)-1]
    max_tup = max(current_col, key = lambda i : i[2])
    current_sentence_tag.append(max_tup[1])

    previous_tag = max_tup[0]
    previous_index = all_tags.index(previous_tag)

    # for other columns, recursively find max pai value and its corresponding tag
    loop_list = list(range(len(each_sentence)-1))
    loop_list.reverse()
    for i in loop_list:
        current_col = pai[i]
        max_tup = current_col[previous_index]
        current_sentence_tag.append(max_tup[1])
        previous_tag = max_tup[0]
        previous_index = all_tags.index(previous_tag)

    # flip the tags since this is a traverse back process
    current_sentence_tag.reverse()

    # save predict tags
    viterbi_result_tag_list = viterbi_result_tag_list+current_sentence_tag


# In[28]:


# add predicted tag as a new column to compute accuracy
dev['viterbi_tag']=viterbi_result_tag_list
num_correct = dev[dev['correct_tag']==dev['viterbi_tag']].iloc[:,0].size
print('The accuracy of applying Viterbi algorithm on dev data is '+str(num_correct/dev.iloc[:,0].size))


# In[29]:


#### predicting tags for test data ####
# load test data
test = pd.read_table('test',header=None, names=['index','word'])
index_list_test = test.values.tolist()


# In[30]:


# because the Viterbi algorithm predict sentence by sentence, so data is splited into list of sentences
def get_sentences(i_list):
    current_sentence = []
    result = []
    for each_pair in i_list:
        if each_pair[0]!=1:
            current_sentence.append(each_pair)
        else:
            result.append(current_sentence)
            current_sentence = [each_pair]
    result = result[1:]
    result.append(current_sentence)
    return result
all_sentences_test = get_sentences(index_list_test)
#all_sentences_test


# In[31]:


#### Viterbi Algorithm ####
viterbi_result_tag_list = []
num_of_tags = len(all_tags)

# for each sentence, compute tags for each word
for each_sentence in all_sentences_test:

    # initialize a Table to keep pai value, each cell contain (previous_tag,current_tag,max(transition*emission))
    # for the first column, the previous_tag is '.'
    pai = []
    previous_tag="."
    each_row = each_sentence[0]
    current_word = each_row[1]

    # if the word is not in word dictionary, we treat it as unknown
    if current_word not in all_word:
        # treat the word as '<unk_digit>' if the word start with a digit
        if current_word[0]>= '0' and current_word[0] <= '9':
            current_word='<unk_digit>'
        else:
            current_word='<unk>'

    # initialize first column of pai based on first word of sentence
    current_col = []
    # for each tag, compute (transition*emission), save to table pai
    for each_tag in all_tags:
        t_value = transition.get((previous_tag,each_tag),0)
        e_value = emission.get((each_tag,current_word),0)
        current_col.append((previous_tag,each_tag,t_value*e_value))
    pai.append(current_col)


    # compute other columns recursively
    for i in range(1,len(each_sentence)):
        each_row = each_sentence[i]
        current_word = each_row[1]
        # if the word is not in word dictionary, we treat it as unknown
        if current_word not in all_word:
            # treat the word as '<unk_digit>' if the word start with a digit
            if current_word[0]>= '0' and current_word[0] <= '9':
                current_word='<unk_digit>'
            else:
                current_word='<unk>'


        previous_col = pai[i-1]
        current_col = []

        # for each current tag, compute (transition*emission)for each previous tag and find argmax of it,
        # then save (previous_tag,current_tag,max(pai*transition*emission) to table pai
        for each_tag in all_tags:
            test_max_list = []
            for each_previous_pair in previous_col:
                each_previous_tag = each_previous_pair[1] # get previous tag
                each_previous_value = each_previous_pair[2]  # get previous pai value
                t_value = transition.get((each_previous_tag,each_tag),0)
                e_value = emission.get((each_tag,current_word),0)
                test_max_list.append(t_value*e_value*each_previous_value)

            # find argmax of (pai*transition*emission)
            max_value = max(test_max_list)
            max_index = test_max_list.index(max_value)
            previous_tag = all_tags[max_index]
            current_col.append((previous_tag,each_tag,max_value))
        pai.append(current_col)



    # traverse back
    current_sentence_tag = []

    # start from last column, find max pai value and its corresponding tag
    current_col = pai[len(each_sentence)-1]
    max_tup = max(current_col, key = lambda i : i[2])
    current_sentence_tag.append(max_tup[1])

    previous_tag = max_tup[0]
    previous_index = all_tags.index(previous_tag)

    # for other columns, recursively find max pai value and its corresponding tag
    loop_list = list(range(len(each_sentence)-1))
    loop_list.reverse()
    for i in loop_list:
        current_col = pai[i]
        max_tup = current_col[previous_index]
        current_sentence_tag.append(max_tup[1])
        previous_tag = max_tup[0]
        previous_index = all_tags.index(previous_tag)

    # flip the tags since this is a traverse back process
    current_sentence_tag.reverse()

    # save predict tags
    viterbi_result_tag_list = viterbi_result_tag_list+current_sentence_tag


# In[32]:


# add predicted tag as a new column and output it
test['viterbi_tag']=viterbi_result_tag_list

# output to viterbi.out
task4_output_list = test.values.tolist()
first_flag = True
with open('viterbi.out', 'w') as f:
    for each_output in task4_output_list:
        index = each_output[0]
        word_type = each_output[1]
        tag = each_output[2]
        if index==1:
            if first_flag:
                first_flag = False
            else:
                f.write('\n')

        f.write(str(index)+'\t'+word_type+'\t'+tag)
        f.write('\n')
f.close()
