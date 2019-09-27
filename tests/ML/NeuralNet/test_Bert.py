import pytest

import JLpyUtils.ML as ML

text_list = ['Marietta Boulevard Northwest', 'Peachtree Street Southwest', 'Mitchell Street Southwest', 'Glenwood Avenue Southeast', 'East Lake Boulevard Southeast', 'Bernice Street', 'Piedmont Road Northeast', 'Dan Lane Northeast', 'Garson Drive', 'North Avenue Northwest', 'Spring Street Northwest', 'Courtland Street Northeast']

def test_word2vect():
    
    Vectorizer = ML.NeuralNet.Bert.word2vect(model_ID = 'bert-base-uncased')
    vectors = [Vectorizer.fit_transform(text) for text in text_list]
    
    assert(len(vectors[0])==768)