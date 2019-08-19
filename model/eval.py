import tensorflow as tf
import numpy as np
import os
import time
import datetime
import operator
import metrics
from collections import defaultdict
from model import data_helpers

# Files
tf.flags.DEFINE_string("test_file", "", "path to test file")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("char_vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("output_file", "", "prediction output file")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_utter_num", 15, "max utterance number")
tf.flags.DEFINE_integer("max_utter_len", 20, "max utterance length")
tf.flags.DEFINE_integer("max_response_num", 20, "max response candidate number")
tf.flags.DEFINE_integer("max_response_len", 20, "max response length")
tf.flags.DEFINE_integer("max_persona_num", 5, "max persona number")
tf.flags.DEFINE_integer("max_persona_len", 15, "max persona length")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

vocab = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))
charVocab = data_helpers.load_char_vocab(FLAGS.char_vocab_file)
print('charVocab size: {}'.format(len(charVocab)))

test_dataset = data_helpers.load_dataset(FLAGS.test_file, vocab, FLAGS.max_utter_num, FLAGS.max_utter_len, FLAGS.max_response_len, FLAGS.max_persona_len)
print('test dataset size: {}'.format(len(test_dataset)))

print("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        utterances = graph.get_operation_by_name("utterances").outputs[0]
        utterances_len = graph.get_operation_by_name("utterances_len").outputs[0]

        responses = graph.get_operation_by_name("responses").outputs[0]  
        responses_len = graph.get_operation_by_name("responses_len").outputs[0]

        utterances_num = graph.get_operation_by_name("utterances_num").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        u_char_feature = graph.get_operation_by_name("utterances_char").outputs[0]
        u_char_len = graph.get_operation_by_name("utterances_char_len").outputs[0]

        r_char_feature = graph.get_operation_by_name("responses_char").outputs[0]
        r_char_len = graph.get_operation_by_name("responses_char_len").outputs[0]

        personas = graph.get_operation_by_name("personas").outputs[0]
        personas_len = graph.get_operation_by_name("personas_len").outputs[0]
        p_char_feature = graph.get_operation_by_name("personas_char").outputs[0]
        p_char_len = graph.get_operation_by_name("personas_char_len").outputs[0]
        personas_num = graph.get_operation_by_name("personas_num").outputs[0]

        # Tensors we want to evaluate
        pred_prob = graph.get_operation_by_name("prediction_layer/prob").outputs[0]

        results = defaultdict(list)
        num_test = 0
        test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, FLAGS.max_utter_num, FLAGS.max_utter_len, \
            FLAGS.max_response_num, FLAGS.max_response_len, FLAGS.max_persona_num, FLAGS.max_persona_len, \
            charVocab, FLAGS.max_word_length, shuffle=False)
        for test_batch in test_batches:
            x_utterances, x_utterances_len, x_response, x_response_len,\
            x_utters_num, x_target, x_ids, \
            x_u_char, x_u_char_len, x_r_char, x_r_char_len, \
            x_personas, x_personas_len, x_p_char, x_p_char_len, x_personas_num = test_batch
            feed_dict = {
                utterances: x_utterances,
                utterances_len: x_utterances_len,
                responses: x_response,
                responses_len: x_response_len,
                utterances_num: x_utters_num,
                dropout_keep_prob: 1.0,
                u_char_feature: x_u_char,
                u_char_len: x_u_char_len,
                r_char_feature: x_r_char,
                r_char_len: x_r_char_len,
                personas: x_personas,
                personas_len: x_personas_len,
                p_char_feature: x_p_char,
                p_char_len: x_p_char_len,
                personas_num: x_personas_num
            }
            predicted_prob = sess.run(pred_prob, feed_dict)
            num_test += len(predicted_prob)
            print('num_test_sample={}'.format(num_test))

            for i in range(len(predicted_prob)):
                probs = predicted_prob[i]
                us_id = x_ids[i]
                label = x_target[i]
                labels = np.zeros(FLAGS.max_response_num)
                labels[label] = 1
                for r_id, prob in enumerate(probs):
                    results[us_id].append((str(r_id), labels[r_id], prob))

accu, precision, recall, f1, loss = metrics.classification_metrics(results)
print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

mvp = metrics.mean_average_precision(results)
mrr = metrics.mean_reciprocal_rank(results)
top_1_precision = metrics.top_1_precision(results)
total_valid_query = metrics.get_num_valid_query(results)
print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

out_path = FLAGS.output_file
print("Saving evaluation to {}".format(out_path))
with open(out_path, 'w') as f:
    f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
    for us_id, v in results.items():
        v.sort(key=operator.itemgetter(2), reverse=True)
        for i, rec in enumerate(v):
            r_id, label, prob_score = rec
            rank = i+1
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(us_id, r_id, prob_score, rank, label))
