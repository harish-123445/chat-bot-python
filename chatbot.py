#import the required libraries

from tokenize import TokenInfo
import numpy as np
import nltk
import string
import random
#nltk.download('omw-1.4')
#importing and reading the corpus(repository of data)

f=open('C:/Users/haris/OneDrive/Desktop/Harish/chat bot in python/chatbot.txt','r',errors='ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower()#converting text to lowercase
nltk.download('punkt') #punkt is a tokenizer

#punkt is the pretrained tokenizer
#punkt is used because it is easy and fantastic to work with it

nltk.download('wordnet') #wordnet is a dictionary
sent_tokens=nltk.sent_tokenize(raw_doc)
word_tokens=nltk.word_tokenize(raw_doc)

#example of sentence tokens
#print(sent_tokens[:2])

#example of word tokens

#print(word_tokens[:2])

#text preprocessing

lemmer=nltk.stem.WordNetLemmatizer()

#wordnet is a semantically-oriented dictionary of english
#included in NLTK

def lemtokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)

def lemnormalize(text):
    return lemtokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#defining a greeeting function

greet_inputs=("hello","hi","greetings","what's up")
greet_response=["hi","hey","hi there","hello","I am glad ! you are talking to me"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_response)

#response generation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo1_response=''
    TfidfVec=TfidfVectorizer(tokenizer=lemnormalize,stop_words='english')
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):#condition if chatbot dont understand whats on screen
        robo1_response=robo1_response + "I am sorry ! I don't understand you"
        return robo1_response
    else:
        robo1_response=robo1_response+sent_tokens[idx]
        return robo1_response

#defining conversation start/end protocol
flag=True
print("BOT : My name is Alexa .Let's have a conversation !\n If you want to end this,type bye")
while(flag==True):
    user_response=input("User :")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks'):
            flag=False
            print("BOT :You are welcome")
        else:
            if(greet(user_response)!=None):
                print("BOT :"+greet(user_response))
            
            else:
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print("BOT :",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("BOT :Good bye !!take care")

#reference 
#great learing youtube channel
