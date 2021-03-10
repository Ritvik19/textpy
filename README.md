# textpy

NLP made Easy

[![Downloads](https://pepy.tech/badge/textpy)](https://pepy.tech/project/textpy)
[![Downloads](https://pepy.tech/badge/textpy/month)](https://pepy.tech/project/textpy)
[![Downloads](https://pepy.tech/badge/textpy/week)](https://pepy.tech/project/textpy)

## Installation

    pip install textpy

or

    pip install git+https://github.com/Ritvik19/textpy.git

## Documentation

### Cleaning

* **Text Cleaner**: A general purpose text cleaning pipeline which utilizes `spacy` and regex to:
  * lower cases the text
  * removes urls and emails
  * removes html css and js
  * removes stop words
  * performs lemmatization
  * removes numbers, punctuations
  * trims white spaces

        textcleaner = cleaning.TextCleaner()
        data['cleaned'] = textcleaner.fit_transform(data['raw'])

### Embeddings

* **BERT Encoder**: A keras layer to learn embeddings from raw text using BERT and finetune it

        text_input = Input(shape=(), dtype=tf.string, name='text')
        net = embeddings.BERTEncoder()(text_input)

        net = layers.Dropout(0.1)(net)
        net = layers.Dense(1, activation="sigmoid", name='classifier')(net)
        classifier_model = models.Model(text_input, net)

* **PreTrained Embeddings**: A keras initializer to fine tune [Pretrained Word Embeddings](https://www.kaggle.com/iezepov/gensim-embeddings-dataset)

        pretrainedEmbeddings = embeddings.PreTrainedEmbeddings(
            'path to embeddings',
            seq_vectorizer=vectorizer,
        )

        inputs = Input(shape=x_train.shape[1])
        x = layers.Embedding(vocab_size, embedding_dimension, embeddings_initializer=pretrainedEmbeddings())(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256)(x)   
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(1, 'sigmoid')(x)

        model = models.Model(inputs, outputs)

### Modelling

* **Rocchio Classifier**: A learner that finds the most Similar Cluster Centroid for a given text
* **Similar Neighbors Classifier**: A learner that finds 'n' most Similar Cluster Centroids for a given text

        # these models work directly on text data fed as a pandas DataFrame
        model = modelling.RocchioClassifier(vectorizer).fit(train[X], train[y])
        # or
        model = modelling.SimilarNeighborsClassifier(vectorizer).fit(train[X], train[y])
        preds = model.predict(valid[X])


### Vectorization

* **Convert To Sparse Tensor**: A utiliy to convert sparse vectors into sparse tensors

        cvt_to_tensors = vectorization.ConvertToSparseTensor()
        sparse_tensors = cvt_to_tensors.fit_transform(sparse_vectors)


* **Sequence Vectorizer**: Vectorize a text corpus, by turning each text into either a sequence of integers of a fixed length

        vectorizer = vectorization.SequenceVectorizer(
            vocab_size=vocab_size, max_seq_len=seq_length
        ).fit(dataset['text'])

* **Word2Vec Vectorizer**: Create Word2Vec Embeddings for a corpus
* **IDF Weighted Word2Vec Vectorizer**: Create IDF weighted Word2Vec Embeddings for a corpus

        vectorizer = vectorization.IdfWeightedWord2VecVectorizer().fit(dataset['text'])
        # or
        vectorizer = vectorization.Word2VecVectorizer().fit(dataset['text'])


## Usage

1. [Fine Tuning BERT](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/BERT.ipynb)
2. [Deep Learning with Embeddings](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Custom-Embeddings-Deep-Learning.ipynb)
3. [Machine Learning with Embeddings](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Custom-Embeddings-Machine-Learning.ipynb)
4. [Deep Learning with Frequency based Vectors](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Frequency-Vectors-Deep-Learning.ipynb)
5. [Machine Learning with Frequency based Vectors](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Frequency-Vectors-Machine-Learning.ipynb)
6. [Fine Tuning Embeddings - Deep Learning](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Pretrained-Embeddinngs-Deep-Learning.ipynb)
7. [Rocchio Classifier](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Rocchio-Classifier.ipynb)
8. [Similar Neighbors Classifier](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Similar-Neighbors-Classifier.ipynb)
9. [Machine Learning with Weighted Embeddings](https://nbviewer.jupyter.org/github/Ritvik19/textpy-doc/blob/main/usage/Weighted-Embeddings-Machine-Learning.ipynb)