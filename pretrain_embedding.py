from gensim.models.word2vec import Word2Vec
import numpy as np
import pickle

def main(dim, src_path, target_path):
    """
    Get word embedding for words in `src_path`
    and output the embedding matrix into `target_path`
    """
    
    train, dev, test, word2index, index2word = pickle.load(open(src_path, 'r'))
    
    #get the training sentences
    sents = [
        [
            index2word[word_ind]
            for word_ind in word_inds
        ]
        for word_inds in train[0].tolist()
    ]
    
    model = Word2Vec(
        size = dim, 
        window = 10, 
        min_count = 3, 
        workers = 4
    )
    

    model.build_vocab(sents)

    model.train(sents)
    
    em = np.zeros((len(model.vocab), dim))

    for w in model.vocab:
        em[word2index[w]] = model[w]

    model.most_similar(positive = ["good"])

    # last row to zero
    em[word2index["<PADDING>"]] = 0

    print em
    em.dump(target_path)


if __name__ == "__main__":
    main(48, 
         'data/stanfordSentimentTreebank/trees/processed.pkl', 
         'data/stanfordSentimentTreebank/trees/pretrained.npy')
