from sklearn.datasets import fetch_20newsgroups
from wordprocess import trun_word
import numpy as np
from utils import LOG_INFO

LOG_INFO("Start!")

newsgroups_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
while "" in newsgroups_train.data:
    newsgroups_train.data.remove("")

LOG_INFO("Data fetched!")

M = len(newsgroups_train.data)  # M = 18846
matwt = [] # (word, topic) matrix

dic = {}
ndocuments = 0
nvoc = 0
ntopics = 20
list_word = []
totalwords = 0

# Generate the dictionary and the list of words in the dictionary
# Also generate the (word, topic) matrix
for idoc in range(M):
    words = newsgroups_train.data[idoc].split()
    indexed = []
    for word in words:
        truned = trun_word(word)
        if truned == "":
            continue
        gentopic = np.random.randint(ntopics)
        totalwords += 1
        if not dic.has_key(truned):
            indexed.append([nvoc, gentopic])
            dic[truned] = nvoc    # Add the new word to the dictionary and list_word
            nvoc += 1
            list_word.append(truned)
        else:            
            indexed.append([dic[truned], gentopic])
    if len(indexed) > 0:
        matwt.append(indexed)
        ndocuments += 1

LOG_INFO("Initial (word, topic) matrix generated!")
print "totalwords:", totalwords

# Begin implementing Collapsed GibbsLDA
mattheta = np.zeros([ndocuments, ntopics]) # For each document, #occurence of topics
matphi = np.zeros([ntopics, nvoc]) # For each topic, #occurence of words

idoc = 0
for doc in matwt:
    for [word, topic] in doc:
        mattheta[idoc, topic] += 1
        matphi[topic, word] += 1
    idoc += 1

alpha = np.ones(ntopics) * 50.0 / ntopics  # priori of theta (document --> topic)
beta = np.ones(nvoc) * 0.01  # priori of phi (topic --> word)
topictotalwords = np.sum(beta) + np.sum(matphi, axis=1)

for epoch in range(100):
    for itera in range(totalwords):
        idoc = np.random.randint(ndocuments)    # It may be better to sample idoc
                                                # proportional to length of documents
        iword = np.random.randint(len(matwt[idoc]))
#    for idoc in range(ndocuments):
#        for iword in range(len(matwt[idoc])):
        [word, topic_before] = matwt[idoc][iword]
        probtheta = alpha + mattheta[idoc, :]
        probphi = (beta[word] + matphi[:, word]) / topictotalwords
        probsample = probtheta * probphi
        probnormed = probsample / float(np.sum(probsample))
        topic_after = np.random.choice(ntopics, p=probnormed)
        mattheta[idoc, topic_before] -= 1
        mattheta[idoc, topic_after] += 1
        matphi[topic_before, word] -= 1
        matphi[topic_after, word] += 1
        topictotalwords[topic_before] -= 1
        topictotalwords[topic_after] += 1
        matwt[idoc][iword][1] = topic_after
    
    LOG_INFO('Training epoch %d results:' % (epoch+1) )
    f = open("result.txt", 'a')
    f.write('Training epoch %d results:\n' % (epoch+1) )
    matrank = np.argsort(matphi, axis = 1)
    for itopic in range(ntopics):
        topic_words = []
        for wordindex in matrank[itopic][-10:]:
            topic_words.append(list_word[wordindex])
        topic_words.reverse()
        print "Topic", itopic, ": ", " ".join(topic_words), \
             "Word count:", topictotalwords[itopic] - np.sum(beta)
        f.write('Topic ' + str(itopic) + ': ' + " ".join(topic_words) +
            " Word count:" + str(topictotalwords[itopic] - np.sum(beta)) + '\n')
    f.close()
    
# Modification1: Add "like", "people", "just", "know", "use", "think", "time", 
# "new", "good", "make", "god", "used", "way", "said", "did", "say", "right",
# "want", "need", "work", "ive" to stop words.

# Modification2: Change alpha to 50.0/ntopics.

# Modification3: Random sampling when choosing coordinates.(Cause the topics be very uniform)
# Did not adopt.

# Modification4: Fix the bug by dividing by topictotalwords