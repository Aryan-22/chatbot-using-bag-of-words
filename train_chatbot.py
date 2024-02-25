import nltk
#nltk.download("punkt")
#nltk.download("wordnet")
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
import random
words = []
classes = []
documents = []
ignore_words = ["?","!"]
data_file = open("intent.json").read()
intents = json.loads(data_file) #json file used as dictionary
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern) #tokenizing each pattern element like "hi","hey" "splits list word by word" "returns list"
        words.extend(w)
        #add documents in the corpus
        documents.append((w,intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
#lemmatize and lower each word and also remove the duplicates... and also do the same with classes list
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words ,classes = sorted(list(set(words))) , sorted(list(set(classes)))
#documents = combination between patterns and tag (w,intents["tag"])
#classes = intents["tags"]
#words = unqiue lemmatized words
print("len of combinations between patterns and tag:",len(documents))
print(documents)
print("no. of intents:",len(classes))
print(classes)
print("no. of unqiue lemmatized words:",len(words))
print(words)

pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))

#creating training data
training = []

#empty array for output
output_empty = [0] * len(classes)

#training set ,bag of words for each sentence
for doc in documents:
    #initializing our bag of words
    bag = []
    pattern_words = doc[0] #list
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words] #creating base words for a collection of words ()
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag,output_row])
print(training)

#shuffle our features and turn into numpy array
random.shuffle(training)
training = np.array(training,dtype=object)
x_train = training[:,0].tolist()
y_train = training[:,1].tolist()
print(x_train)
print(y_train)
model = Sequential()
model.add(Dense(128,input_shape = (len(x_train[0]),),activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation = "softmax"))

#using sgd optimizer
sgd = SGD(lr = 0.01,decay = 1e-6,momentum = 0.9,nesterov = True)
print(model.summary())
model.compile(loss = "categorical_crossentropy",optimizer = sgd,metrics = ["accuracy"])

#fitting and saving the model
hist = model.fit(x_train,y_train,epochs = 200,batch_size = 5,verbose = 1)
model.save("chatbot_model.h5",hist)
print("model created!")










