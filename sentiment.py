from __future__ import division
import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    train_pos_neg=train_pos+train_neg
    allwords=[item for sublist in train_pos_neg for item in sublist]
    word_set=set(allwords)
    
    print "word_set: ", len(word_set) 
    f_set=word_set-stopwords
    print "f_set: ", len(f_set) 

    word_set=set(f_set)
    
    len_p=len(train_pos)
    len_n=len(train_neg)

    dict_pos=dict()
    dict_neg=dict()

    for line in train_pos:
	for word in set(line):
		if word in dict_pos:
			dict_pos[word]=dict_pos[word]+1
		else:
			dict_pos.update({word:1})
	
    for line in train_neg:
	for word in set(line):
		if word in dict_neg:
			dict_neg[word]=dict_neg[word]+1
		else:
			dict_neg.update({word:0})

    for word in list(f_set):
	freq_pos=0.0
	freq_neg=0.0

	if word in dict_pos:
		freq_pos=dict_pos[word]

	if word in dict_neg:
		freq_neg=dict_neg[word]

	if freq_pos<0.01*len_p and freq_neg<0.01*len_n:
		f_set.discard(word)
	else:
		if  freq_pos<2*freq_neg and freq_neg<2*freq_pos:
			f_set.discard(word)

    print len(f_set)
    f_list=list(f_set)
    f_len=len(f_set)
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec=[]
    for line in train_pos:
	temp=[0]*f_len
	for word in set(line):
		if word in f_set:
			temp[f_list.index(word)]=1
        train_pos_vec.append(temp)

    print "Train_pos_vec created!"

    train_neg_vec=[]
    for line in train_neg:
	temp=[0]*f_len
	for word in set(line):
		if word in f_set:
			temp[f_list.index(word)]=1
        train_neg_vec.append(temp)

    test_pos_vec=[]
    for line in test_pos:
	temp=[0]*f_len
	for word in set(line):
		if word in f_set:
			temp[f_list.index(word)]=1
        test_pos_vec.append(temp)

    test_neg_vec=[]
    for line in test_neg:
	temp=[0]*f_len
	for word in set(line):
		if word in f_set:
			temp[f_list.index(word)]=1
        test_neg_vec.append(temp)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos=[]
    i=0
    for line in train_pos:
	labeled_train_pos.append(LabeledSentence(line, ['train_pos_{}'.format(i)]))
	i=i+1

    labeled_train_neg=[]
    i=0
    for line in train_neg:
	labeled_train_neg.append(LabeledSentence(line, ['train_neg_{}'.format(i)]))
	i=i+1

    labeled_test_pos=[]
    i=0
    for line in test_pos:
	labeled_test_pos.append(LabeledSentence(line, ['test_pos_{}'.format(i)]))
	i=i+1

    labeled_test_neg=[]
    i=0
    for line in test_neg:
	labeled_test_neg.append(LabeledSentence(line, ['test_neg_{}'.format(i)]))
	i=i+1
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)
    
    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    
    train_pos_vec=[]
    train_neg_vec=[]
    test_pos_vec=[]
    test_neg_vec=[]

    for line in labeled_train_pos:
	train_pos_vec.append(model.docvecs[line.tags[0]])

    for line in labeled_train_neg:
	train_neg_vec.append(model.docvecs[line.tags[0]])

    for line in labeled_test_pos:
	test_pos_vec.append(model.docvecs[line.tags[0]])

    for line in labeled_test_neg:
	test_neg_vec.append(model.docvecs[line.tags[0]])

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model= BernoulliNB()
    nb_model.fit(train_pos_vec+train_neg_vec, Y)    
    BernoulliNB(alpha=1.0, binarize=None, class_prior=None, fit_prior=True)

    lr_model=LogisticRegression()
    lr_model.fit(train_pos_vec+train_neg_vec, Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model=GaussianNB()
    nb_model.fit(train_pos_vec+train_neg_vec, Y)


    lr_model=LogisticRegression()
    lr_model.fit(train_pos_vec+train_neg_vec, Y)
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    
    tp=model.predict(test_pos_vec).tolist().count('pos')
    fn=model.predict(test_pos_vec).tolist().count('neg')
    tn=model.predict(test_neg_vec).tolist().count('neg')
    fp=model.predict(test_neg_vec).tolist().count('pos')

    accuracy=(tp+tn)/(tp+tn+fn+fp)
    #tn=model.predict(test_neg_vec)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
