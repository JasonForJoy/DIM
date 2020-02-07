import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")

def get_char_embedding(charVocab):
    print("get_char_embedding")
    char_size = len(charVocab)
    embeddings = np.zeros((char_size, char_size), dtype='float32')
    for i in range(1, char_size):
        embeddings[i, i] = 1.0
    return tf.constant(embeddings, name="word_char_embedding")

def load_embed_vectors(fname, dim):
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec
    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embedded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        #else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 
    return embeddings 


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states

def cnn_layer(inputs, filter_sizes, num_filters, scope=None, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse):
        input_size = inputs.get_shape()[2].value

        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_{}".format(i)):
                w = tf.get_variable("w", [filter_size, input_size, num_filters])
                b = tf.get_variable("b", [num_filters])
            conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
            pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
            outputs.append(pooled)
    return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]


def attended_response(similarity_matrix, contexts, flattened_utters_len, max_utter_len, max_utter_num):
    # similarity_matrix:    [batch_size, max_response_num, max_response_len, max_utter_num*max_utter_len]
    # contexts:             [batch_size, max_utter_num*max_utter_len, dim]
    # flattened_utters_len: [batch_size* max_utter_num, ]
    max_response_num = similarity_matrix.get_shape()[1].value
    
    # masked similarity_matrix
    mask_c = tf.sequence_mask(flattened_utters_len, max_utter_len, dtype=tf.float32)  # [batch_size*max_utter_num, max_utter_len]
    mask_c = tf.reshape(mask_c, [-1, max_utter_num*max_utter_len])                    # [batch_size, max_utter_num*max_utter_len]
    mask_c = tf.expand_dims(mask_c, 1)                                                # [batch_size, 1, max_utter_num*max_utter_len]
    mask_c = tf.expand_dims(mask_c, 2)                                                # [batch_size, 1, 1, max_utter_num*max_utter_len]
    similarity_matrix = similarity_matrix * mask_c + -1e9 * (1-mask_c)                # [batch_size, max_response_num, response_len, max_utter_num*max_utter_len]

    attention_weight_for_c = tf.nn.softmax(similarity_matrix, dim=-1)                 # [batch_size, max_response_num, response_len, max_utter_num*max_utter_len]
    contexts_tiled = tf.tile(tf.expand_dims(contexts, 1), [1, max_response_num, 1, 1])# [batch_size, max_response_num, max_utter_num*max_utter_len, dim]
    attended_response = tf.matmul(attention_weight_for_c, contexts_tiled)             # [batch_size, max_response_num, response_len, dim]

    return attended_response

def attended_context(similarity_matrix, responses, flattened_responses_len, max_response_len, max_response_num):
    # similarity_matrix:       [batch_size, max_response_num, max_response_len, max_utter_num*max_utter_len]
    # responses:               [batch_size, max_response_num, max_response_len, dim]
    # flattened_responses_len: [batch_size* max_response_num, ]

    # masked similarity_matrix
    mask_r = tf.sequence_mask(flattened_responses_len, max_response_len, dtype=tf.float32)  # [batch_size*max_response_num, max_response_len]
    mask_r = tf.reshape(mask_r, [-1, max_response_num, max_response_len])  # [batch_size, max_response_num, max_response_len]
    mask_r = tf.expand_dims(mask_r, -1)                                    # [batch_size, max_response_num, max_response_len, 1]
    similarity_matrix = similarity_matrix * mask_r + -1e9 * (1-mask_r)     # [batch_size, max_response_num, max_response_len, max_utter_num*max_utter_len]

    attention_weight_for_r = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0,1,3,2]), dim=-1)  # [batch_size, max_response_num, max_utter_num*max_utter_len, response_len]
    attended_context = tf.matmul(attention_weight_for_r, responses)                                  # [batch_size, max_response_num, max_utter_num*max_utter_len, dim]
    
    return attended_context


class DIM(object):
    def __init__(
      self, max_utter_num, max_utter_len, max_response_num, max_response_len, max_persona_num, max_persona_len, 
        vocab_size, embedding_size, vocab, rnn_size, maxWordLength, charVocab, l2_reg_lambda=0.0):

        self.utterances = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances")
        self.utterances_len = tf.placeholder(tf.int32, [None, max_utter_num], name="utterances_len")
        self.utters_num = tf.placeholder(tf.int32, [None], name="utterances_num")

        self.responses = tf.placeholder(tf.int32, [None, max_response_num, max_response_len], name="responses")
        self.responses_len = tf.placeholder(tf.int32, [None, max_response_num], name="responses_len")

        self.personas = tf.placeholder(tf.int32, [None, max_persona_num, max_persona_len], name="personas")
        self.personas_len = tf.placeholder(tf.int32, [None, max_persona_num], name="personas_len")
        self.personas_num = tf.placeholder(tf.int32, [None], name="personas_num")
        
        self.target = tf.placeholder(tf.int64, [None], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.u_charVec = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len, maxWordLength], name="utterances_char")
        self.u_charLen = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances_char_len")

        self.r_charVec = tf.placeholder(tf.int32, [None, max_response_num, max_response_len, maxWordLength], name="responses_char")
        self.r_charLen =  tf.placeholder(tf.int32, [None, max_response_num, max_response_len], name="responses_char_len")

        self.p_charVec = tf.placeholder(tf.int32, [None, max_persona_num, max_persona_len, maxWordLength], name="personas_char")
        self.p_charLen =  tf.placeholder(tf.int32, [None, max_persona_num, max_persona_len], name="personas_char_len")

        l2_loss = tf.constant(1.0)

        # =============================== Embedding layer ===============================
        # word embedding
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            utterances_embedded = tf.nn.embedding_lookup(W, self.utterances)  # [batch_size, max_utter_num, max_utter_len, word_dim]
            responses_embedded = tf.nn.embedding_lookup(W, self.responses)    # [batch_size, max_response_num, max_response_len, word_dim]
            personas_embedded = tf.nn.embedding_lookup(W, self.personas)      # [batch_size, max_persona_num, max_persona_len, word_dim]
            print("original utterances_embedded: {}".format(utterances_embedded.get_shape()))
            print("original responses_embedded: {}".format(responses_embedded.get_shape()))
            print("original personas_embedded: {}".format(personas_embedded.get_shape()))
        
        with tf.name_scope('char_embedding'):
            char_W = get_char_embedding(charVocab)
            utterances_char_embedded = tf.nn.embedding_lookup(char_W, self.u_charVec)  # [batch_size, max_utter_num, max_utter_len,  maxWordLength, char_dim]
            responses_char_embedded = tf.nn.embedding_lookup(char_W, self.r_charVec)   # [batch_size, max_response_num, max_response_len, maxWordLength, char_dim]
            personas_char_embedded = tf.nn.embedding_lookup(char_W, self.p_charVec)    # [batch_size, max_persona_num, max_persona_len, maxWordLength, char_dim]
            print("utterances_char_embedded: {}".format(utterances_char_embedded.get_shape()))
            print("responses_char_embedded: {}".format(responses_char_embedded.get_shape()))
            print("personas_char_embedded: {}".format(personas_char_embedded.get_shape()))

        char_dim = utterances_char_embedded.get_shape()[-1].value
        utterances_char_embedded = tf.reshape(utterances_char_embedded, [-1, maxWordLength, char_dim])  # [batch_size*max_utter_num*max_utter_len, maxWordLength, char_dim]
        responses_char_embedded = tf.reshape(responses_char_embedded, [-1, maxWordLength, char_dim])    # [batch_size*max_response_num*max_response_len, maxWordLength, char_dim]
        personas_char_embedded = tf.reshape(personas_char_embedded, [-1, maxWordLength, char_dim])      # [batch_size*max_persona_num*max_persona_len, maxWordLength, char_dim]

        # char embedding
        utterances_cnn_char_emb = cnn_layer(utterances_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=False) # [batch_size*max_utter_num*max_utter_len, emb]
        cnn_char_dim = utterances_cnn_char_emb.get_shape()[1].value
        utterances_cnn_char_emb = tf.reshape(utterances_cnn_char_emb, [-1, max_utter_num, max_utter_len, cnn_char_dim])                                # [batch_size, max_utter_num, max_utter_len, emb]

        responses_cnn_char_emb = cnn_layer(responses_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=True)    # [batch_size*max_response_num*max_response_len,  emb]
        responses_cnn_char_emb = tf.reshape(responses_cnn_char_emb, [-1, max_response_num, max_response_len, cnn_char_dim])                            # [batch_size, max_response_num, max_response_len, emb]

        personas_cnn_char_emb = cnn_layer(personas_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=True)      # [batch_size*max_persona_num*max_persona_len,  emb]
        personas_cnn_char_emb = tf.reshape(personas_cnn_char_emb, [-1, max_persona_num, max_persona_len, cnn_char_dim])                                # [batch_size, max_persona_num, max_persona_len, emb]
                
        utterances_embedded = tf.concat(axis=-1, values=[utterances_embedded, utterances_cnn_char_emb])   # [batch_size, max_utter_num, max_utter_len, emb]
        responses_embedded  = tf.concat(axis=-1, values=[responses_embedded, responses_cnn_char_emb])     # [batch_size, max_response_num, max_response_len, emb]
        personas_embedded  = tf.concat(axis=-1, values=[personas_embedded, personas_cnn_char_emb])        # [batch_size, max_persona_num, max_persona_len, emb]
        utterances_embedded = tf.nn.dropout(utterances_embedded, keep_prob=self.dropout_keep_prob)
        responses_embedded = tf.nn.dropout(responses_embedded, keep_prob=self.dropout_keep_prob)
        personas_embedded = tf.nn.dropout(personas_embedded, keep_prob=self.dropout_keep_prob)
        print("utterances_embedded: {}".format(utterances_embedded.get_shape()))
        print("responses_embedded: {}".format(responses_embedded.get_shape()))
        print("personas_embedded: {}".format(personas_embedded.get_shape()))


        # =============================== Encoding layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            
            emb_dim = utterances_embedded.get_shape()[-1].value
            flattened_utterances_embedded = tf.reshape(utterances_embedded, [-1, max_utter_len, emb_dim])  # [batch_size*max_utter_num, max_utter_len, emb]
            flattened_utterances_len = tf.reshape(self.utterances_len, [-1])                               # [batch_size*max_utter_num, ]
            flattened_responses_embedded = tf.reshape(responses_embedded, [-1, max_response_len, emb_dim]) # [batch_size*max_response_num, max_response_len, emb]
            flattened_responses_len = tf.reshape(self.responses_len, [-1])                                 # [batch_size*max_response_num, ]
            flattened_personas_embedded = tf.reshape(personas_embedded, [-1, max_persona_len, emb_dim])    # [batch_size*max_persona_num, max_persona_len, emb]
            flattened_personas_len = tf.reshape(self.personas_len, [-1])                                   # [batch_size*max_persona_num, ]

            rnn_scope_name = "bidirectional_rnn"
            u_rnn_output, u_rnn_states = lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)
            utterances_output = tf.concat(axis=2, values=u_rnn_output)  # [batch_size*max_utter_num, max_utter_len, rnn_size*2]
            r_rnn_output, r_rnn_states = lstm_layer(flattened_responses_embedded, flattened_responses_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)
            responses_output = tf.concat(axis=2, values=r_rnn_output)   # [batch_size*max_response_num, max_response_len, rnn_size*2]
            p_rnn_output, p_rnn_states = lstm_layer(flattened_personas_embedded, flattened_personas_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)
            personas_output = tf.concat(axis=2, values=p_rnn_output)    # [batch_size*max_persona_num, max_persona_len, rnn_size*2]
            print("encoded utterances : {}".format(utterances_output.shape))
            print("encoded responses : {}".format(responses_output.shape))
            print("encoded personas : {}".format(personas_output.shape))


        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:

            output_dim = utterances_output.get_shape()[-1].value
            utterances_output = tf.reshape(utterances_output, [-1, max_utter_num*max_utter_len, output_dim])      # [batch_size, max_utter_num*max_utter_len, rnn_size*2]
            utterances_output_tiled = tf.tile(tf.expand_dims(utterances_output, 1), [1, max_response_num, 1, 1])  # [batch_size, max_response_num, max_utter_num*max_utter_len, rnn_size*2]
            responses_output = tf.reshape(responses_output, [-1, max_response_num, max_response_len, output_dim]) # [batch_size, max_response_num, max_response_len, rnn_size*2]
            personas_output = tf.reshape(personas_output, [-1, max_persona_num*max_persona_len, output_dim])      # [batch_size, max_persona_num*max_persona_len, rnn_size*2]
            personas_output_tiled = tf.tile(tf.expand_dims(personas_output, 1), [1, max_response_num, 1, 1])      # [batch_size, max_response_num, max_persona_num*max_persona_len, rnn_size*2]
            
            # 1. cross-attention between context and response
            similarity_UR = tf.matmul(responses_output,  # [batch_size, max_response_num, response_len, max_utter_num*max_utter_len]
                                      tf.transpose(utterances_output_tiled, perm=[0,1,3,2]), name='similarity_matrix_UR')
            attended_utterances_output_ur = attended_context(similarity_UR, responses_output, flattened_responses_len, max_response_len, max_response_num)  # [batch_size, max_response_num, max_utter_num*max_utter_len, dim]
            attended_responses_output_ur = attended_response(similarity_UR, utterances_output, flattened_utterances_len, max_utter_len, max_utter_num)       # [batch_size, max_response_num, response_len, dim]
            
            m_u_ur = tf.concat(axis=-1, values=[utterances_output_tiled, attended_utterances_output_ur, tf.multiply(utterances_output_tiled, attended_utterances_output_ur), utterances_output_tiled-attended_utterances_output_ur])  # [batch_size, max_response_num, max_utter_num*max_utter_len, dim]
            m_r_ur = tf.concat(axis=-1, values=[responses_output, attended_responses_output_ur, tf.multiply(responses_output, attended_responses_output_ur), responses_output-attended_responses_output_ur])                         # [batch_size, max_response_num, response_len, dim]
            concat_dim = m_u_ur.get_shape()[-1].value
            m_u_ur = tf.reshape(m_u_ur, [-1, max_utter_len, concat_dim])    # [batch_size*max_response_num*max_utter_num, max_utter_len, dim]
            m_r_ur = tf.reshape(m_r_ur, [-1, max_response_len, concat_dim]) # [batch_size*max_response_num, max_response_len, dim]
            
            rnn_scope_cross = 'bidirectional_rnn_cross'
            rnn_size_layer_2 = rnn_size
            tiled_flattened_utterances_len = tf.reshape(tf.tile(tf.expand_dims(self.utterances_len, 1), [1, max_response_num, 1]), [-1, ]) # [batch_size*max_response_num*max_utter_num, ]
            u_ur_rnn_output, u_ur_rnn_state = lstm_layer(m_u_ur, tiled_flattened_utterances_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=False)
            r_ur_rnn_output, r_ur_rnn_state = lstm_layer(m_r_ur, flattened_responses_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)
            utterances_output_cross_ur = tf.concat(axis=-1, values=u_ur_rnn_output)   # [batch_size*max_response_num*max_utter_num, max_utter_len, rnn_size*2]
            responses_output_cross_ur = tf.concat(axis=-1, values=r_ur_rnn_output)    # [batch_size*max_response_num, max_response_len, rnn_size*2]
            print("establish cross-attention between context and response")


            # 2. cross-attention between persona and response without decay
            similarity_PR = tf.matmul(responses_output,  # [batch_size, max_response_num, response_len, max_persona_num*max_persona_len]
                                      tf.transpose(personas_output_tiled, perm=[0,1,3,2]), name='similarity_matrix_PR')
            attended_personas_output_pr = attended_context(similarity_PR, responses_output, flattened_responses_len, max_response_len, max_response_num) # [batch_size, max_response_num, max_persona_num*max_persona_len, dim]
            attended_responses_output_pr = attended_response(similarity_PR, personas_output, flattened_personas_len, max_persona_len, max_persona_num)   # [batch_size, max_response_num, response_len, dim]

            m_p_pr = tf.concat(axis=-1, values=[personas_output_tiled, attended_personas_output_pr, tf.multiply(personas_output_tiled, attended_personas_output_pr), personas_output_tiled-attended_personas_output_pr])  # [batch_size, max_response_num, max_persona_num*max_persona_len, dim]
            m_r_pr = tf.concat(axis=-1, values=[responses_output, attended_responses_output_pr, tf.multiply(responses_output, attended_responses_output_pr), responses_output-attended_responses_output_pr])           # [batch_size, max_response_num, response_len, dim]
            m_p_pr = tf.reshape(m_p_pr, [-1, max_persona_len, concat_dim])   # [batch_size*max_response_num*max_persona_num, max_persona_len, dim]
            m_r_pr = tf.reshape(m_r_pr, [-1, max_response_len, concat_dim])  # [batch_size*max_response_num, max_response_len, dim]
            
            tiled_flattened_personas_len = tf.reshape(tf.tile(tf.expand_dims(self.personas_len, 1), [1, max_response_num, 1]), [-1, ]) # [batch_size*max_response_num*max_persona_num, ]
            p_pr_rnn_output, p_pr_rnn_state = lstm_layer(m_p_pr, tiled_flattened_personas_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)
            r_pr_rnn_output, r_pr_rnn_state = lstm_layer(m_r_pr, flattened_responses_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)
            personas_output_cross_pr = tf.concat(axis=-1, values=p_pr_rnn_output)   # [batch_size*max_response_num*max_persona_num, max_persona_len, rnn_size*2]
            responses_output_cross_pr = tf.concat(axis=-1, values=r_pr_rnn_output)  # [batch_size*max_response_num, max_response_len, rnn_size*2]
            print("establish cross-attention between persona and response")


        # =============================== Aggregation layer ===============================
        with tf.variable_scope("aggregation_layer") as vs:
            # aggregate utterance across utterance_len
            final_utterances_max = tf.reduce_max(utterances_output_cross_ur, axis=1)
            final_utterances_state = tf.concat(axis=1, values=[u_ur_rnn_state[0].h, u_ur_rnn_state[1].h])
            final_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])  # [batch_size*max_response_num*max_utter_num, 4*rnn_size]

            # aggregate utterance across utterance_num
            final_utterances = tf.reshape(final_utterances, [-1, max_utter_num, output_dim*2])   # [batch_size*max_response_num, max_utter_num, 4*rnn_size]
            tiled_utters_num = tf.reshape(tf.tile(tf.expand_dims(self.utters_num, 1), [1, max_response_num]), [-1, ])  # [batch_size*max_response_num, ]
            rnn_scope_aggre = "bidirectional_rnn_aggregation"
            final_utterances_output, final_utterances_state = lstm_layer(final_utterances, tiled_utters_num, rnn_size, self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=False)
            final_utterances_output = tf.concat(axis=2, values=final_utterances_output)  # [batch_size*max_response_num, max_utter_num, 2*rnn_size]
            final_utterances_max = tf.reduce_max(final_utterances_output, axis=1)                                          # [batch_size*max_response_num, 2*rnn_size]
            final_utterances_state = tf.concat(axis=1, values=[final_utterances_state[0].h, final_utterances_state[1].h])  # [batch_size*max_response_num, 2*rnn_size]
            aggregated_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])               # [batch_size*max_response_num, 4*rnn_size]

            # aggregate response across response_len
            final_responses_max = tf.reduce_max(responses_output_cross_ur, axis=1)                           # [batch_size*max_response_num, 2*rnn_size]
            final_responses_state = tf.concat(axis=1, values=[r_ur_rnn_state[0].h, r_ur_rnn_state[1].h])     # [batch_size*max_response_num, 2*rnn_size]
            aggregated_responses_ur = tf.concat(axis=1, values=[final_responses_max, final_responses_state]) # [batch_size*max_response_num, 4*rnn_size]
            print("establish RNN aggregation on context and response")


            # aggregate persona across persona_len
            final_personas_max = tf.reduce_max(personas_output_cross_pr, axis=1)                        # [batch_size*max_response_num*max_persona_num, 2*rnn_size]
            final_personas_state = tf.concat(axis=1, values=[p_pr_rnn_state[0].h, p_pr_rnn_state[1].h]) # [batch_size*max_response_num*max_persona_num, 2*rnn_size]
            final_personas = tf.concat(axis=1, values=[final_personas_max, final_personas_state])       # [batch_size*max_response_num*max_persona_num, 4*rnn_size]

            # aggregate persona across persona_num
            # 1. RNN aggregation
            # final_personas = tf.reshape(final_personas, [-1, max_persona_num, output_dim*2])   # [batch_size*max_response_num, max_persona_num, 4*rnn_size]
            # tiled_personas_num = tf.reshape(tf.tile(tf.expand_dims(self.personas_num, 1), [1, max_response_num]), [-1, ])  # [batch_size*max_response_num, ]
            # final_personas_output, final_personas_state = lstm_layer(final_personas, tiled_personas_num, rnn_size, self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=True)
            # final_personas_output = tf.concat(axis=2, values=final_personas_output)  # [batch_size*max_response_num, max_persona_num, 2*rnn_size]
            # final_personas_max = tf.reduce_max(final_personas_output, axis=1)                                        # [batch_size*max_response_num, 2*rnn_size]
            # final_personas_state = tf.concat(axis=1, values=[final_personas_state[0].h, final_personas_state[1].h])  # [batch_size*max_response_num, 2*rnn_size]
            # aggregated_personas = tf.concat(axis=1, values=[final_personas_max, final_personas_state])               # [batch_size*max_response_num, 4*rnn_size]
            # print("establish RNN aggregation on persona")
            # 2. ATT aggregation
            final_personas = tf.reshape(final_personas, [-1, max_persona_num, output_dim*2])                               # [batch_size*max_response_num, max_persona_num, 4*rnn_size]
            pers_w = tf.get_variable("pers_w", [output_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
            pers_b = tf.get_variable("pers_b", shape=[1, ], initializer=tf.zeros_initializer())
            pers_weights = tf.nn.relu(tf.einsum('aij,jk->aik', final_personas, pers_w) + pers_b)                           # [batch_size*max_response_num, max_persona_num, 1]
            tiled_personas_num = tf.reshape(tf.tile(tf.expand_dims(self.personas_num, 1), [1, max_response_num]), [-1, ])  # [batch_size*max_response_num, ]
            mask_p = tf.expand_dims(tf.sequence_mask(tiled_personas_num, max_persona_num, dtype=tf.float32), -1)           # [batch_size*max_response_num, max_persona_num, 1]
            pers_weights = pers_weights * mask_p + -1e9 * (1-mask_p)                                                       # [batch_size*max_response_num, max_persona_num, 1]
            pers_weights = tf.nn.softmax(pers_weights, dim=1)
            aggregated_personas = tf.matmul(tf.transpose(pers_weights, [0, 2, 1]), final_personas)  # [batch_size*max_response_num, 1, 4*rnn_size]
            aggregated_personas = tf.squeeze(aggregated_personas, [1])                              # [batch_size*max_response_num, 4*rnn_size]

            # aggregate response across response_len
            final_responses_max = tf.reduce_max(responses_output_cross_pr, axis=1)                           # [batch_size*max_response_num, 2*rnn_size]
            final_responses_state = tf.concat(axis=1, values=[r_pr_rnn_state[0].h, r_pr_rnn_state[1].h])     # [batch_size*max_response_num, 2*rnn_size]
            aggregated_responses_pr = tf.concat(axis=1, values=[final_responses_max, final_responses_state]) # [batch_size*max_response_num, 4*rnn_size]
            print("establish ATT aggregation on persona and response")

            joined_feature =  tf.concat(axis=1, values=[aggregated_utterances, aggregated_responses_ur, aggregated_personas, aggregated_responses_pr])  # [batch_size*max_response_num, 16*rnn_size(3200)]
            print("joined feature: {}".format(joined_feature.get_shape()))
            

        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            hidden_input_size = joined_feature.get_shape()[1].value
            hidden_output_size = 256
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            #regularizer = None
            # dropout On MLP
            joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                            activation_fn=tf.nn.relu,
                                                            reuse=False,
                                                            trainable=True,
                                                            scope="projected_layer") # [batch_size*max_response_num, hidden_output_size(256)]
            full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)

            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.reshape(tf.matmul(full_out, s_w) + bias, [-1, max_response_num])   # [batch_size, max_response_num]
            print("logits: {}".format(logits.get_shape()))
            
            self.probs = tf.nn.softmax(logits, name="prob")  # [batch_size, max_response_num]

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.probs, 1), self.target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
