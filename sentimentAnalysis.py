
# coding: utf-8

# In[230]:

import numpy as np
import tensorflow as tf
from collections import Counter


# In[231]:

# Read files

with open('reviews.txt', 'r') as f:
    reviews = f.read()


# In[232]:

reviews[:200]


# # Noise Reduction

# In[233]:

from string import punctuation
all_words = ''.join([character for character in reviews if character not in punctuation])
reviews = all_words.split('\n')


# In[234]:

all_words = ' '.join(reviews)
reviews_text = all_words.split()


# In[193]:

reviews_text[:200]


# In[5]:

#reviews_text[:660]


# In[235]:

bag_of_words = Counter(reviews_text)


# In[236]:

bag_of_words


# In[196]:

bag_of_words.most_common()


# In[237]:

unique_words = sorted(bag_of_words, key=bag_of_words.get, reverse=True)


# In[238]:

unique_words


# In[239]:

unique_words[12767]


# In[240]:

unique_words[64]


# In[241]:

# Starting one index off for padding with a 0
word_to_index = {word: i for i, word in enumerate(unique_words, 1)}


# In[242]:

word_to_index['rickshaws']


# In[243]:

# Convert each of our reviews into their corresponding incides (of our unique words)
reviews_ints = []
for each in reviews:
    reviews_ints.append([word_to_index[word] for word in each.split()])


# In[244]:

word_to_index['bromwell']


# In[245]:

reviews_ints[0][0]


# # Encoding our labels

# In[246]:

with open('labels.txt', 'r') as f:
    labels = f.read()
labels = labels.split("\n")


# In[247]:

labels_ints = [1 if label == 'positive' else 0 for label in labels]
labels_ints


# In[210]:

#labels_ints = np.array(labels_ints)


# In[248]:

# Review length counts
review_lengths = Counter([len(x) for x in reviews_ints])


# In[249]:

# Checking if there are any 0 length reviews
review_lengths[0]


# In[250]:

non_zero_review_indexes = [index for index, review in enumerate(reviews_ints) if len(review) != 0]


# In[251]:

len(non_zero_review_indexes)


# In[252]:

reviews_ints[-1]


# In[253]:

reviews_ints = [reviews_ints[index] for index in non_zero_review_indexes]


# In[256]:

labels = np.array([labels_ints[index] for index in non_zero_review_indexes])


# In[257]:

labels


# In[258]:

# The max length of a review is too many steps for the RNN
# Let's cut the reviews into segments of 200 steps
# If they're shorter than 200, we'll pad them from the left

max_length = 200
zero_matrix = np.zeros((len(reviews_ints), max_length), dtype = int)


# In[259]:

# Fill in the zero matrix with the values corresponding to the reviews
for index, row in enumerate(reviews_ints):
    # zero_matrix[current review, the last n values corresponding to the reviews size] = the first 200 of that review 
    zero_matrix[index, -len(row):] = np.array(row)[:max_length]


# In[260]:

zero_matrix


# In[261]:

data = zero_matrix
data[:10,:100]


# In[262]:

data[185, 94]


# # Data Splitting

# In[277]:

# Training & Validation Sets - 0.8, 0.1
training_validation_ratio = 0.8
split_index = int(len(data) * training_validation_ratio)
train_x, val_x = data[: split_index], data[split_index : ]
train_y, val_y = labels[: split_index], labels[split_index :]

# Test Sets
test_validation_ratio = 0.5
test_index = int(len(val_x) * test_validation_ratio)
val_x, test_x = val_x[ : test_index], val_x[test_index : ]
val_y, test_y = val_y[ : test_index], val_y[test_index : ]

train_x.shape


# In[278]:

val_x.shape


# In[279]:

test_x.shape


# In[289]:

# RESET GRAPH For Continuous Testing
#tf.reset_default_graph()


# # Model

# In[290]:

# Hyper-Parameters

# units in the hidden layer in LSTM cells
lstm_size = 128

# number of stacked lstm layers
lstm_layers = 1

batch_size = 500
learning_rate = 0.001

# TF Graph elements
model = tf.Graph()
# Dropout probabiliity is a scalar
with model.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name = "inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name = "labels")
    dropout_probability = tf.placeholder(tf.float32, name = "dropout_probability")


# In[291]:

len(word_to_index)


# In[292]:

# Embeddings Layer (Lookup Table from Word2Vec Models)
embedded_layer_size = 300

with model.as_default():
    embedding_matrix = tf.Variable(tf.random_uniform((len(word_to_index), embedded_layer_size), -1, 1))
    embedding_vectors = tf.nn.embedding_lookup(embedding_matrix, inputs_)
    embedding_aggregated = tf.reduce_sum(embedding_vectors, [1])


# In[293]:

# LSTM Cell Layer

with model.as_default():
    
    # Basic LSTM
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Apply Dropout
    dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = dropout_probability)
    
    # Stack up the LSTM cells
    cell = tf.contrib.rnn.MultiRNNCell([dropout] * lstm_layers)
    
    # Get Initial Zero State
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    # Forward Pass
    outputs, final_state = tf.nn.dynamic_rnn(cell, embedding_vectors, initial_state = initial_state)


# In[294]:

# Output

with model.as_default():
    # We only care about the last output in outputs for the sentiment prediction
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn = tf.sigmoid)
    error = tf.losses.mean_squared_error(labels_, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)


# In[295]:

# Validation

# Additional nodes to calculate the accuracy for validation forward pass
with model.as_default():
    correct_prediction = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[296]:

# Creating the actual batches

def get_batches(x, y, batch_size = 100):
    number_batches = len(x) // batch_size
    x, y = x[: number_batches * batch_size], y[: number_batches * batch_size]
    for index in range(0, len(x), batch_size):
        yield x[index : index + batch_size], y[index : index + batch_size]


# In[297]:

# Training

epochs = 10

with model.as_default():
    saver = tf.train.Saver()
    

with tf.Session(graph = model) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    
    for epoch in range(epochs):
        current_state = sess.run(initial_state)
        
        for index, (x,y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_ : x, 
                    labels_ : y[:, None], 
                    dropout_probability : 0.5, 
                    initial_state : current_state}
            loss, current_state, _ = sess.run([error, final_state, optimizer], feed_dict = feed)
            
            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(epoch, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))
    
            if iteration % 25 == 0:
                # Validation
                validation_accuracy = []
                validation_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {training_inputs : x, training_labels : y[:, None], dropout_probability : 1, initial_state : validation_state}
                    batch_accuracy, validation_state = sess.run([accuracy, final_state], feed_dict = feed)
                    validation_accuracy.append(batch_accuracy)
                print("The Validation Accuracy : {:.3f}".format(np.mean(validation_accuracy)))
            
            iteration += 1

    saver.save(sess, "checkpoints/sentiment.ckpt")

    


# # Testing

# In[ ]:

test_accuracy = []
with tf.Session(graph = model) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for x, y in get_batches(test_x, test_y, batch_size):
        feed = {inputs_ : x,
                label_ : y[:, None],
                dropout_probability: 1,
                initial_state: test_state}
        batch_accuracy, test_state = sess.run([accuracy, final_state], feed_dict = feed)
        test_accuracy.append(batch_accuracy)
    print("The test accuracy is : {:.3f}".format(np.mean(test_accuracy)))
        

