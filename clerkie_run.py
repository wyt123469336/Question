
# coding: utf-8

# In[1]:


#Author: Yitao Wu
#coding challange
#Clerkie

import nltk
import json
import numpy as np
import csv
import answer_questions
from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import wordnet
stemmer = SnowballStemmer("english")


# In[2]:


# load calculated words classes and vector values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    words = np.asarray(synapse['words'])
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])
    classes = np.asarray(synapse['classes'])
word_token = set(words)


# In[3]:


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is the bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

# error threshold
ERROR_THRESHOLD = 0.6
def classify(sentence, show_details=False):
    
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ('[class, %]:',return_results)
    return return_results


# In[4]:


# classify("How much money do I have in my bank accounts?")
# classify("How much money can I spare?")
# classify("Can I buy this $2.3M apartment?")
# classify("Can I manage to purchase this $1.6M house?")
# classify("What's the total amount of money in my accounts?")
#classify("Can I buy a $2M ferrari")
#classify("I would love to check my bank balance", show_details=True)
#classify("check budget")


# In[5]:


# add related yet not recognized question to file, for future study
def add_question(user_input,class_name):
    f = open("untrained.txt", "a")
    f.write("{},{}\n".format(class_name,user_input))
    f.close()
    print("Thanks! I've saved your question for future learning :D")


# In[6]:


# capture entities in questions belong to category 1.
# extract bank names, checking/saving, account number
def cap_entities(user_input):
    token = nltk.word_tokenize(user_input)
    tagged = nltk.pos_tag(token)
    bank_set = set(['boa','bankofamerica','chase','citi'])
    bank_id_type = ['','','']
    potential_bank_name = ''
    
    for i,x in enumerate(tagged):
#         if x[0] == 'in':
#             is_my = 0
#             try:
#                 if tagged[i+1][0] == 'my':
#                     is_my = 1
#                 potential_bank_name = ''.join(token[i+1+is_my:])
#             except:
#                 print('bank name not provided')
        if 'NN' in x[1] and x[0] in bank_set:
            bank_id_type[0] = x[0]
        elif x[1] == 'CD':
            bank_id_type[1] = x[0]
        elif x[0] == 'checking' or x[0] == 'saving' or x[0] == 'credit':
            bank_id_type[2] = x[0]

    return bank_id_type


# capture entities in questions belong to category 3. 
# determine if the user is asking about buying houses
# return price only if user specified a price
# and the question is about buying house
def house_keyword(user_input):
    token = nltk.word_tokenize(user_input)
    tagged = nltk.pos_tag(token)
    price = '0'
    flag = False
    similarity_threshold = 0.8
    house_set = ['house',
                 'condo',
                 'pad',
                 'crib',
                 'apartment',
                 'residence',
                 'mansion'
                ]
    for x in tagged:
        if x[1] == 'CD':
            price = x[0]
        if 'NN' in x[1]:
            try:
                w1 = wordnet.synset('{}.n.01'.format(x[0]))
            except:
                continue
            for word in house_set:
                w2 = wordnet.synset('{}.n.01'.format(word))
                if w1.wup_similarity(w2) > similarity_threshold:
                    flag = True
                    break
    return price if flag else '-1'
        


# In[7]:


#is user question relatable?
related_threshold = 0.4


#Main function
def ask_clerkie():
    print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    user_input = input("what's your financial question? (type 'quit' to exit.) \n\n").lower()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    if 'hello' in user_input:
        print('Hi Yitao :D, do you have a financial question?')
        return True
    #quitting outer loop
    elif user_input == 'quit':
        print('Thanks for using clerkie :D')
        print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        return False
    
    #if user input nothing
    elif len(user_input) == 0:
        print('Ask Me a question :D')
        return True
    user_token = clean_up_sentence(user_input)
    related_token = 0
    
    #check how many tokens are related to financial questions
    #also calculate the related percentage
    for token in user_token:
        if token in word_token:
            related_token += 1
    
    #related question, and question length more than 2 words
    if related_token/len(user_token) >= related_threshold and len(user_token) >= 2:
        question_class = classify(user_input)
        
        #this means the question is not recognized by Clerkie
        #ask user's help to label the question
        #and save to file for future learning
        if len(question_class) == 0:
            print("sorry, I'm still learning :p, care to classify the question for me?")
            print("0: check balance\n1:check budget:\n2:check affordable\n3:other categories")
            class_name = input("0 or 1 or 2 or 3? (press enter to skip)\n")
            
            #record question and its label
            if class_name in set(['0','1','2','3']):
                add_question(user_input,class_name)
                
        
        
        #question that classified as category 1
        elif question_class[0][0] == 0:
            
            bank_id_type = cap_entities(user_input)
            if bank_id_type[0] == 'bankofamerica':
                bank_id_type[0] = 'boa'
            answer_questions.get_balance(bank_id_type)
            
        #question that classified as category 2
        elif question_class[0][0] == 1:
            answer_questions.get_budget()
            
        #question that classified as category 3
        elif question_class[0][0] == 2:
            price = house_keyword(user_input)
            
            #question about buying things other than house
            if price == '-1':
                print("sorry, I can only help with house affordability :p")
            
            #user didn't specify a price in the question
            elif price == '0':
                print("sorry, please provide house price so I can help :p")
            else:
                answer_questions.is_affordable(price)
        
    else:
        print("Please ask financial related question (more than 2 words) :p")
    return True
    
    


# In[8]:


flag = True
while flag:
    flag = ask_clerkie()
    

