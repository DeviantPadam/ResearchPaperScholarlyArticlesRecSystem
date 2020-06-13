#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% [markdown]
# # ***Scholarly Articles Recommender Engine Using Doc2Vec***

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [markdown]
# > #### I have tried to make a recommendation engine using Doc2Vec using ArXiv research papers meta-data dataset and text of tags descreption is taken from [arxiv.org.].
# > #### Suggestions are most welcomed that can improve the recommendations.

# %% [code]
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser,Phrases
from time import time
import multiprocessing
from gensim.matutils import Dense2Corpus
from gensim.similarities import MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models import LdaModel,KeyedVectors
import umap
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary

# %% [markdown]
# > **Data Cleaning**

# %% [code]
df = pd.read_json('../input/arxivdataset/arxivData.json',orient='records')
df.head()

# %% [code]
df2 = pd.DataFrame(df.author.str.split('}').tolist(),index = df.index).stack()
df2.head()

# %% [code]
df3 = pd.DataFrame(df.link.str.split(', ').tolist(),index = df.index).stack()

# %% [code]
def rem_unwanted(line):
    return re.sub("\'term'|\'rel'|\'href'|\'type'|\'title'|\[|\{|\'name'|\'|\]|\,|\}",'',line).strip(' ').strip("''").strip(":")

# %% [code]
df2 = pd.DataFrame(df2.apply(rem_unwanted))

# %% [code]
df2 = pd.DataFrame(df2.unstack().iloc[:,0:2].to_records()).drop(columns={'index'})

# %% [code]
df2.columns = ['Author1','Author2']

# %% [code]
df2.Author1 = df2.Author1.str.strip(' ')
df2.Author2 = df2.Author2.str.strip(' ')

# %% [code]
df2[df2.Author2 == '']

# %% [code]
df2 = df2.reset_index().drop(columns='index')

# %% [code]
df2.head()

# %% [code]
len(df2.Author1.unique())

# %% [code]
df3 = pd.DataFrame(df3.apply(rem_unwanted,convert_dtype=True))

# %% [code]
df3 = df3.unstack()

# %% [code]
links = df3.iloc[:,[1,4]]
links.columns = ['textLink','pdfLink']
links.head()

# %% [code]
tags = pd.DataFrame(df['tag'].str.split(',').tolist())
# tags = tags[tags.str.contains('term')]
# tags.head()
tags = tags.iloc[:,[0,3,6]].stack()
# tags = tags.iloc[:,0]

# %% [code]
tags

# %% [code]
tags = tags.apply(rem_unwanted)

# %% [code]
tags = tags.unstack()

# %% [code]
tags[0] = tags[0].str.strip()
tags.iloc[:,1] = tags.iloc[:,1].str.strip()
tags.iloc[:,2] = tags.iloc[:,2].str.strip()

# %% [code]
tags.columns = ['Topic1','Topic2','Topic3']

# %% [code]
tags

# %% [code]
pre0 = pd.merge(df,tags,how = 'inner',left_index=True,right_index=True).drop('tag',axis=1)
pre = pd.merge(pre0,df2,how = 'inner',left_index=True,right_index=True).drop('author',axis=1)
data = pd.merge(pre,links,how = 'inner',left_index=True,right_index=True).drop('link',axis=1)

# %% [code]
def rem_bracket(line):
#     return re.sub("\'term'|\'rel'|\'href'|\'type'|\'title'|\[|\{|\'name'|\'|\]|\)}",'',line).strip(' ').strip("''").strip(":")
    return line.strip(')')

# %% [markdown]
# > Initial Data

# %% [code]
df.head()

# %% [markdown]
# > Cleaned Data

# %% [code]
data.head()

# %% [code]
data[data['Author2']=='']

# %% [markdown]
# > Cleaning the topics text file

# %% [code]
tags = pd.read_csv('../input/arxivtagsdescription/tags.txt',sep='/n',header=None,engine='python')

# %% [code]
tags.head()

# %% [code]
d1 = pd.DataFrame(tags.iloc[[i for i in tags.index if i%2==0]].reset_index().iloc[0:47][0].str.split(' - ').tolist())
d2 = pd.DataFrame(tags.iloc[[i for i in tags.index if i%2==0]].reset_index().iloc[47:][0].str.split('(').tolist())

# %% [code]
d1.head()

# %% [code]
d2.head()

# %% [code]
d2[1] = d2[1].apply(rem_bracket)

# %% [code]
d2 = d2.set_index([1]).reset_index()

# %% [code]
d2.columns = [0,1]

# %% [code]
d2.head()

# %% [code]
d3 = pd.concat([d1,d2])

# %% [code]
d3 = d3.reset_index().drop(columns=['index'])

# %% [code]
d3['TopicExplain'] = tags.iloc[[i for i in tags.index if i%2!=0]][0].reset_index()[0]

# %% [code]
d3.columns = ['Topic','FullTopic','TopicExplain']

# %% [code]
tags = d3.copy()

# %% [markdown]
# > Cleaned Topics

# %% [code]
tags.head()

# %% [markdown]
# > Loading spacy for tokenising.

# %% [code]
nlp = spacy.load('en',disable = ['ner','parser'])
# spacy.require_gpu()

# %% [code]
stopwords = list(STOP_WORDS)+list((''.join(string.punctuation)).strip(''))+['-pron-','-PRON-']
len(stopwords)

# %% [code]
def lemmatizer(df):
    texts = []
    c=0
    for text in df:
        if c%1000==0:
            print(c)
        c+=1
        doc = nlp(text)
        lemma = [word.lemma_.lower().strip('') for word in doc]
        words = [word for word in lemma if word not in stopwords]
        texts.append(' '.join(words))
    return pd.Series(texts)

# %% [code]
data.fillna(' ',inplace=True)

# %% [code]
data.isna().any()

# %% [code]
data['summary'][0]

# %% [code]
data['title'][0]

# %% [code]
def rem_n(line):
    return re.sub('\\n',' ',line)

# %% [code]
data['summary'] = data['summary'].apply(rem_n)
data['title'] = data['title'].apply(rem_n)

# %% [code]
data['summary'][0]

# %% [code]
db1 = pd.merge(data,tags,how='left',left_on='Topic1',right_on='Topic').drop(columns=['Topic'])
db2 = pd.merge(db1,tags,how='left',left_on='Topic2',right_on='Topic').drop(columns=['Topic','TopicExplain_y'])
db3 = pd.merge(db2,tags,how='left',left_on='Topic3',right_on='Topic').drop(columns=['Topic','TopicExplain'])

# %% [code]
db3 = db3[['id', 'summary', 'title', 'year', 'FullTopic_x', 'FullTopic_y', 'FullTopic','TopicExplain_x', 'Topic1', 'Topic2', 'Topic3','Author1', 'Author2', 'textLink', 'pdfLink']]

# %% [code]
db3.columns = ['id', 'summary', 'title', 'year','Topic1',
       'Topic2', 'Topic3', 'Topic', 'DTopic1', 'DTopic2', 'DTopic3',
       'Author1', 'Author2', 'textLink', 'pdfLink' ]

# %% [code]
db3.drop(columns=['DTopic1','DTopic2','DTopic3'],inplace=True)

# %% [code]
f1 = db3.copy()

# %% [code]
f1.fillna(' ',inplace=True)

# %% [code]
f1.isna().sum()

# %% [markdown]
# > Concat all text for tokenising

# %% [code]
f1['Full'] = (f1['title']+" "+f1['summary']+' '+f1['Topic1']+' '+f1['Topic2']+' '+f1['Topic3']+' '+f1['Topic']+' '+f1['Author1']+' '+f1['Author2'])

# %% [code]
t = time()
processed_text = lemmatizer(f1['Full'])

# %% [code]
(time()-t)/60

# %% [code]
processed_text.index = f1['id'].values

# %% [code]
processed_text = pd.DataFrame(processed_text)

# %% [code]
processed_text.iloc[0:12].index

# %% [code]
processed_text.iloc[:6].values

# %% [markdown]
# > Creating Bigrams and Trigrams

# %% [code]
phr = [i[0].split() for i in processed_text.values]

# %% [code]
phrases = Phrases(phr,min_count=20,progress_per=1000)

# %% [code]
bigram = Phraser(phrases)

# %% [code]
sentences = bigram[phr]

# %% [code]
phrases = Phrases(sentences,min_count=10,progress_per=1000)

# %% [code]
trigram = Phraser(phrases)

# %% [code]
trigrams = trigram[sentences]

# %% [code]
trigrams[123]

# %% [markdown]
# > Creating Tagged data for Doc2Vec (tag = articleId)

# %% [code]
tagged_data = [TaggedDocument(words=' '.join(i),tags=[j]) for i, j in zip(trigrams,processed_text.index)]

# %% [code]
tagged_data[2]

# %% [code]
docmodel = Doc2Vec(dm=1,vector_size = 300,window=3,workers=4,negative=8,min_count=20)

# %% [code]
docmodel.build_vocab(tagged_data)

# %% [code]
docmodel.corpus_total_words

# %% [code]
docmodel.epochs

# %% [code]
t = time()
docmodel.train(tagged_data,total_examples=docmodel.corpus_count,epochs=20)

# %% [code]
(time()-t)/60

# %% [code]
docmodel.init_sims(replace=True)
docmodel.save('model')

# %% [code]
docmodel = Doc2Vec.load('model')

# %% [code]
 docmodel.docvecs.most_similar('1802.00209v1')

# %% [code]
# f2 = f1[['id','title','summary','FullTopic','TopicExplain','Author1','Author2']]

# %% [code]
def get_recommendations(*n):
    j = docmodel.docvecs.most_similar(positive=n)
    r = f1[f1['id'].isin(list(n))]
    p = ['Searched',]*len(r)
    for i in j:
        r = pd.concat([r,f1[f1['id']==i[0]]])
        p.append(i[1])
#     r = f2[f2['id'].isin(a)]
    r['ProbabilityChance'] = p
    return r

# %% [code]
rec = get_recommendations('1802.00209v1')
# ('1305.3814v2')

# %% [code]
rec

# %% [code]
mat = TfidfVectorizer().fit_transform(lemmatizer(rec['Full']))

# %% [markdown]
# > Similarity in Recommended Articles

# %% [code]
(cosine_similarity(mat,mat)*100)[:,0]

# %% [markdown]
# ### ***Some Drawbacks - ***
# * ### As I chose window size very less because many topics contains very little text. See below that topic contains literally no data so prediction will be hard.

# %% [code]
tags[tags['FullTopic']=='Mathematical Software']['TopicExplain'].values

# %% [markdown]
# * ### Data is uneven. Mostly articles have AI as their primary or sub-topic. 

# %% [markdown]
# # **Suggestions are heartily welcome.**


# In[ ]:




