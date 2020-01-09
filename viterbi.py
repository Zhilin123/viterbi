#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 22:43:19 2020

@author: wangxiujiang
"""
import numpy as np
import nltk


a = [
     [0.2767,0.0006,0.0031,0.0453,0.0449,0.0510,0.2026],
     [0.3777,0.0110,0.0009,0.0084,0.0584,0.009,0.0025],
     [0.0008,0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041],
     [0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231],
     [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036],
     [0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068],
     [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
     [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]
     ]

rows = ["start","NNP","MD","VB","JJ","NN","RB","DT"]
cols = rows[1:]
words = ["Janet","will","back","the", "bill"]

b = [
     [0.000032, 0, 0, 0.000048, 0],
     [0, 0.308431, 0, 0, 0],
     [0, 0.000028, 0.000672, 0, 0.000028],
     [0, 0, 0.000340, 0, 0],
     [0, 0.000200, 0.000223, 0, 0.002337],
     [0, 0, 0.010446, 0, 0],
     [0, 0, 0, 0.506099, 0]
    ]

a = np.array(a)
b = np.array(b)

def viterbi(a,b,beta=None):
    """
    big O time complexity = N**2 *T
    space = N*T
    
    beam search beta* N * T
    """
    N,T = b.shape
    log_viterbi_state = np.zeros((N,T))
    best_path_pointer = np.zeros((N,T))
    # initialize each possibility at first time step to probability of start of each POS tag * probability of word condition on each POS tag
    for i in range(N):
        log_viterbi_state[i,0] = np.log(a[0,i]) + np.log(b[i,0])
        #print(log_viterbi_state[i,0])
    
    #for each time step ie number of tokens
    
    for j in range(1,T):
        # for each POS tag
        for k in range(N):
            if not beta:
            #slightly vectorized form
                possible_viterbi_pathes_to_one_state = log_viterbi_state[:,j-1] + np.log(a[1:,k]) + np.log(b[k,j]) 
            else:
                
                #non-vectorized form to prevent caculation of -inf terms for beam search
                non_zero_elements_in_previous = set(log_viterbi_state[:,j-1][log_viterbi_state[:,j-1] != float("-inf")])
                possible_viterbi_pathes_to_one_state = []
                for k1 in range(N):
                    if log_viterbi_state[k1,j-1] in non_zero_elements_in_previous:
                        value = log_viterbi_state[k1,j-1] + np.log(a[k1+1,k]) + np.log(b[k,j])
                    else:
                        #print("found")
                        value = float("-inf")
                    possible_viterbi_pathes_to_one_state.append(value)
                    
                possible_viterbi_pathes_to_one_state = np.array(possible_viterbi_pathes_to_one_state)
            
                                        # respectively
                                        # likelihood of POS tags of previous token
                                        # transitionary probability of from each POS tag to a specific POS tag (others come later in the for loop)
                                        # probability of word conditional on that POS tag
            #print(np.log(a[1:,k]))
            #print(possible_viterbi_pathes_to_one_state)
            log_viterbi_state[k,j] = np.max(possible_viterbi_pathes_to_one_state)
            best_path_pointer[k,j] = np.argmax(possible_viterbi_pathes_to_one_state)
        
        if beta:
            #beam search
            all_states = log_viterbi_state[:,j]
            sorted_states = np.argsort(all_states)[::-1]
            top_n = sorted_states[:beta]
            for k in range(N):
                if k not in top_n:
                    log_viterbi_state[k,j] = float("-inf")
                
        
    
    best_ending_prob = np.max(log_viterbi_state[:,T-1])
    best_ending_state = np.argmax(log_viterbi_state[:,T-1])
    bestpath = [best_ending_state]
    #get the order
    state = best_ending_state
    for l in range(T-1,0,-1):
        bestpath.append(int(best_path_pointer[state,l]))
        state = int(best_path_pointer[state,l])
    
    bestpath = bestpath[::-1]
    #print(log_viterbi_state)
    return bestpath, best_ending_prob

"""
bestpath, best_ending_prob = viterbi(a,b,beta=2)

j = 0
for i in bestpath:
    print(words[j], " / ", cols[i])
    j += 1
"""

some_text = "The flight was operated by Ukraine International Airlines, the flag carrier and the largest airline of Ukraine, on a scheduled flight from Iranian capital Tehran's Imam Khomeini International Airport to Boryspil International Airport in the Ukrainian capital Kiev. Emergency officials confirmed that the plane was carrying 176 people on board of which 15 were children, and 9 crew members.[7] Flight 752 was scheduled to take off at 05:15 local time (UTC+3:30), however, it was delayed by approximately an hour. It took off from Tehran at 06:12 local time and was expected to land in Kiev at 08:00 local time (UTC+2:00).[8][2] Flight data from Flightradar24 shows no discrepancies in speed or altitude data from what a normal flight would display.[9] The final ADS-B data received was at 06:14, less than 3 minutes after departure. According to the data, the last recorded altitude was at 7,925 feet (2,416 m) above mean sea level with a groundspeed of 275 knots (509 km/h).[10][11] The airport itself is 3,305 feet (1,007 m) above mean sea level, which would give an altitude of 4,620 feet (1,410 m) above ground level. The flight was climbing when the altitude record abruptly ended.[10][12] The aircraft crashed into terrain located 15 kilometres (9.3 mi; 8.1 nmi) North of the airport. A video, circulated on social media, purportedly shows the moment of the crash. The video suggested that the plane was on fire when it began to dive, with some of its parts breaking up in mid-air.[13] It then crashed and exploded.[1] Iranian Students News Agency (ISNA) did not confirm the authenticity of the video, but it did state that the plane was burning prior to the crash, leading to speculation of a possible shootdown.[14][15] Some aviation experts considered it too early to discuss causes. However, aviation monitoring group OPS group stated: We would recommend the starting assumption to be that this was a shootdown event, similar to MH17 – until there is clear evidence to the contrary asserting that photographs show obvious projectile holes in the fuselage and a wing section.[16] Shortly after the crash, emergency responders arrived with 22 ambulances, four bus ambulances, and a helicopter, but heavy fires prevented a rescue attempt. The wreckage was strewn over a wide area, with no survivors found at the crash site centered about 35°33′40″ N, 51°06′14″ E.[17] The plane was obliterated upon impact."


processed_text = ""
for i in some_text:
    if i != " " and not i.isalpha():
        pass
    else:
        processed_text += i.lower()
processed_text = processed_text.split()

def delete_interpolation(corpus):
# purpose is to account for different patterns in the train corpus and the test corpus

    corpus = processed_text
    trigrams = {}
    bigrams = {}
    unigrams = {}
    
    for i in range(len(processed_text)-2):
        if (processed_text[i],processed_text[i+1],processed_text[i+2]) not in trigrams:
            trigrams[(processed_text[i],processed_text[i+1],processed_text[i+2])] = 1
        else:
            trigrams[(processed_text[i],processed_text[i+1],processed_text[i+2])] += 1
        if (processed_text[i+0],processed_text[i+1]) not in bigrams:
            bigrams[(processed_text[i+0],processed_text[i+1])] = 1
        else:
            bigrams[(processed_text[i+0],processed_text[i+1])] += 1
        if processed_text[i+0] not in bigrams:
            unigrams[processed_text[i+0]] = 1
        else:
            unigrams[processed_text[i+0]] += 1
    
    lam1 = 0
    lam2 = 0
    lam3 = 0
    
    for i in trigrams:
        try:
            tri_value = (trigrams[i] - 1) / (bigrams[(i[0],i[1])] - 1)
        except:
            tri_value = 0
        try:
            bi_value = (bigrams[(i[1],i[2])] - 1) / (unigrams[i[1]] - 1)
        except:
            bi_value = 0
        try:
            uni_value = (unigrams[i[2]] - 1)/ (len(unigrams) - 1)
        except:
            uni_value = 0
        result = np.argmax([tri_value,bi_value,uni_value])
        if result == 0:
            lam1 += trigrams[i]
        elif result == 1:
            lam2 += trigrams[i]
        else:
            lam3 += trigrams[i]
    
    total = lam1 + lam2 + lam3
    
    lam1 = lam1 / total
    lam2 = lam2 / total
    lam3 = lam3 / total
    
    return lam1, lam2, lam3


# OOV

#Use letter information to predict tags

from nltk.corpus import treebank
from collections import Counter
import random

tagged_words = treebank.tagged_words()
tagged_sentences = treebank.tagged_sents()

cut_off = int(len(tagged_sentences)*0.8)
train = tagged_sentences[:cut_off]
test = tagged_sentences[cut_off:]
train_words = tagged_words[:80000]
test_words = tagged_words[80000:]



def unigram_tagger(train,test):
    word_to_all_pos = {}
    
    for word, pos in train:
        if word not in word_to_all_pos:
            word_to_all_pos[word] = [pos]
        else:
            word_to_all_pos[word].append(pos)
            
    word_to_pos_freq = {}
    
    for word in word_to_all_pos:
        word_to_pos_freq[word] = Counter(word_to_all_pos[word]).most_common(1)[0][0]
        
    score = 0
    wrong = 0
    wrong_words = []
    for word, pos in test:
        if word in word_to_pos_freq:
            if pos == word_to_pos_freq[word]:
                score += 1
            else:
                wrong += 1
        else:
            contains_digit = False
            for i in word:
                if i.isdigit():
                    contains_digit = True
            if word[0].isupper() and word[-1] == "s":
                prediction = "NNPS"
            elif word[0].isupper():
                prediction = "NNP"
            elif contains_digit:
                prediction = "CD"
            elif word[-1] == "d":
                prediction = "VBD"
            elif "-" in word:
                prediction == "JJ"
            elif word[-1] == "s":
                prediction = "NNS"
            else:
                prediction = "NN"
            if pos == prediction:
                score += 1
            else:
                wrong += 1
                wrong_words.append((word,pos))
            
    print("Most frequent tag conditional on word")
    print("score", score)
    print("wrong", wrong)
    print(score/(score+wrong))
    #print(wrong_words[:50])
    # 92%

unigram_tagger(train_words,test_words)


#data prep
    
#a is the transitionary probability

#def data_prep():
initial_pos_tags = []
all_words = []
for sentence in train:
    for pair in sentence:
        initial_pos_tags.append(pair[1])
        all_words.append(pair[0])

all_pos_tags = list(set(initial_pos_tags))
all_words = list(set(all_words))

pos_to_value = {}
for i in range(len(all_pos_tags)):
    pos_to_value[all_pos_tags[i]] = i + 1

word_to_value = {}
for i in range(len(all_words)):
    word_to_value[all_words[i]] = i 

a = np.zeros((len(all_pos_tags)+1, len(all_pos_tags)))

for sentence in train:
    for word_order in range(len(sentence)):
        if word_order == 0:
            start = 0
        else:
            pos_tag = sentence[word_order-1][1]
            start = pos_to_value[pos_tag]
        end_pos_tag = sentence[word_order][1]
        end = pos_to_value[end_pos_tag] -1
        a[start,end] += 1
        
row_sums = a.sum(axis=1)
a = a / row_sums[:, np.newaxis]

b = np.zeros((len(all_pos_tags),len(all_words)))

for sentence in train:
    for word_order in range(len(sentence)):
        word = sentence[word_order][0]
        pos_tag = sentence[word_order][1]
        col = word_to_value[word]
        row = pos_to_value[pos_tag] - 1
        b[row,col] += 1

row_sums = b.sum(axis=1)
b = b / row_sums[:, np.newaxis]
    
oov = []


#oov

#find p(t|last few 10 letters)


def get_prob_of_n_letters_to_pos_freq(n):
    
    last_10_letters_to_pos = {}
    
    for sentence in train:
        for pair in sentence:
            pos = pair[1]
            word = pair[0]
            if len(word) < n:
                pad = ' ' * (n-len(word))
                word = pad + word
            else:
                word = word[-n:]
            if word not in last_10_letters_to_pos:
                last_10_letters_to_pos[word] =[pos]
            else:
                last_10_letters_to_pos[word].append(pos)
    
    last_10_letters_to_pos_freq = {}
    
    for word in last_10_letters_to_pos:
        total = 0
        counter = Counter(last_10_letters_to_pos[word])
        for i in counter:
            total += counter[i]
        counter_dict = dict(counter)
        for i in counter_dict:
            counter_dict[i] = counter_dict[i] / total
        last_10_letters_to_pos_freq[word] = counter_dict
    
    prob_of_10_letters = {}
    
    for i in last_10_letters_to_pos:
        prob_of_10_letters[i] = len(last_10_letters_to_pos[i]) / len(initial_pos_tags)
        
    prob_of_each_letter = {}

    for i in last_10_letters_to_pos:
        for letter in i:
            if letter not in prob_of_each_letter:
                prob_of_each_letter[letter] = 1
            else:
                prob_of_each_letter[letter] += 1
    total_letters = 0 
    for i in prob_of_each_letter:
        total_letters += prob_of_each_letter[i]
        #prob_of_each_letter[i] = prob_of_each_letter[i] #/ len(initial_pos_tags)
    
    for i in prob_of_each_letter:
        prob_of_each_letter[i] = prob_of_each_letter[i] /total_letters
    return last_10_letters_to_pos_freq, prob_of_10_letters, prob_of_each_letter

all_tables = {}
for n in range(1,11):
    last_10_letters_to_pos_freq, prob_of_10_letters, prob_of_each_letter = get_prob_of_n_letters_to_pos_freq(n)
    all_tables[n] = {
                "last_10_letters_to_pos_freq":last_10_letters_to_pos_freq,
                "prob_of_10_letters":prob_of_10_letters,
                "prob_of_each_letter":prob_of_each_letter
            }


prob_of_tag = dict(Counter(initial_pos_tags))
total_tags = len(initial_pos_tags)
for i in prob_of_tag:
    prob_of_tag[i] = prob_of_tag[i]/total_tags

prob_of_tag_np = np.zeros((len(prob_of_tag),))

for i in prob_of_tag:
    number = pos_to_value[i] - 1
    prob_of_tag_np[number] = prob_of_tag[i]

### find emission of oov words
    
word = 'Zhilin'

def oov_emission(word):
    if len(word) < 10:
        pad = ' ' * (10-len(word))
        word = pad + word
    for n in range(10,0,-1):
        # backoff strategy
        subword = word[-n:]
        last_10_letters_to_pos_freq = all_tables[n]["last_10_letters_to_pos_freq"]
        prob_of_10_letters = all_tables[n]["prob_of_10_letters"]
        #prob_of_each_letter = all_tables[n]["prob_of_each_letter"]
        if subword in last_10_letters_to_pos_freq:
            #calculate the prob of w given tag using naive bayes and backoff
            
            prob_tag_given_word = last_10_letters_to_pos_freq[subword]
            
            prob_tag_given_word_np = np.zeros((len(all_pos_tags),)) 
            for tag in prob_tag_given_word:
               number = pos_to_value[tag] - 1
               prob_tag_given_word_np[number] = prob_tag_given_word[tag]
            prob_word_given_tag = prob_tag_given_word_np * prob_of_10_letters[subword] * (0.4)**(10-n) / prob_of_tag_np
            return prob_word_given_tag
        else:
            continue
    return None

###
    
def give_sentence_specific_b(tagged_sentence, b):
    new_b = []
    correct_pos = []
    for word_pos in tagged_sentence:
        word = word_pos[0]
        pos = word_pos[1]
        if word in word_to_value:
            word_value = word_to_value[word]
            word_vector = b[:,word_value]
        else:
            #considering oov
            #raise Exception('oov')
            #86.5 % considering OOV
            #problem lies with OOV --> 93.6% achieved without that
            oov.append(word)
            word_vector = oov_emission(word)
            #word_vector = np.ones((b.shape[0],)) / row_sums
        new_b.append(word_vector)
        pos_value = pos_to_value[pos] - 1 # pos_value is 1-indexed due to start symbol
        correct_pos.append(pos_value)
        
    new_b = np.array(new_b).T
    path, prob = viterbi(a,new_b)
    matched = 0
    for i in range(len(path)):
        if path[i] == correct_pos[i]:
            matched += 1
    return path, correct_pos, matched, len(path)

def hmm():
    score = 0
    total = 0
    predicted = []
    true = []
    for tagged_sentence in test:
        try:
            path, correct_pos, correct, over = give_sentence_specific_b(tagged_sentence, b)
        except:
            pass
        score += correct
        total += over
        predicted += path
        true += correct_pos
    print("Bigram HMM")
    print(score)
    print(total)
    print(score/total)

hmm()

confusion_matrix(true,predicted)


    


#b is the probability of word given tag

#HMM works well for train but not test compared to bigram why?