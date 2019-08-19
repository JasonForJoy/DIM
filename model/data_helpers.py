import numpy as np
import random


def load_vocab(fname):
    '''
    vocab = {"I": 0, ...}
    '''
    vocab={}
    with open(fname, 'rt') as f:
        for i,line in enumerate(f):
            word = line.decode('utf-8').strip()
            vocab[word] = i
    return vocab

def load_char_vocab(fname):
    '''
    charVocab = {"U": 0, "!": 1, ...}
    '''
    charVocab={}
    with open(fname, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            char_id = int(fields[0])
            ch = fields[1]
            charVocab[ch] = char_id
    return charVocab

def to_vec(tokens, vocab, maxlen):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["fiance"])  # fix to fiance
    return length, np.array(vec)

def load_dataset(fname, vocab, max_utter_num, max_utter_len, max_response_len, max_persona_len):

    dataset=[]
    with open(fname, 'rt') as f:
        for us_id, line in enumerate(f):
            line = line.decode('utf-8').strip()
            fields = line.split('\t')

            # context utterances
            context = fields[0]
            utterances = (context + " ").split(' _eos_ ')[:-1]
            utterances = [utterance + " _eos_" for utterance in utterances]
            utterances = utterances[-max_utter_num:]   # select the last max_utter_num utterances
            us_tokens = []
            us_vec = []
            us_len = []
            for utterance in utterances:
                u_tokens = utterance.split(' ')[:max_utter_len]  # select the head max_utter_len tokens in every utterance
                u_len, u_vec = to_vec(u_tokens, vocab, max_utter_len)
                us_tokens.append(u_tokens)
                us_vec.append(u_vec)
                us_len.append(u_len)
            us_num = len(utterances)
            
            # responses
            responses = fields[1].split("|")
            rs_tokens = []
            rs_vec = []
            rs_len = []
            for response in responses:
                r_tokens = response.split(' ')[:max_response_len]  # select the head max_response_len tokens in every candidate
                r_len, r_vec = to_vec(r_tokens, vocab, max_response_len)
                rs_tokens.append(r_tokens)
                rs_vec.append(r_vec)
                rs_len.append(r_len)

            # label
            label = int(fields[2])

            # other persona
            if fields[3] != "NA" and fields[4] == "NA":
                personas = fields[3].split("|")
                ps_tokens = []
                ps_vec = []
                ps_len = []
                for persona in personas:
                    p_tokens = persona.split(' ')[:max_persona_len]  # select the head max_persona_len tokens in every persona
                    p_len, p_vec = to_vec(p_tokens, vocab, max_persona_len)
                    ps_tokens.append(p_tokens)
                    ps_vec.append(p_vec)
                    ps_len.append(p_len)
                ps_num = len(personas)

            # self persona
            if fields[3] == "NA" and fields[4] != "NA":
                personas = fields[4].split("|")
                ps_tokens = []
                ps_vec = []
                ps_len = []
                for persona in personas:
                    p_tokens = persona.split(' ')[:max_persona_len]  # select the head max_persona_len tokens in every persona
                    p_len, p_vec = to_vec(p_tokens, vocab, max_persona_len)
                    ps_tokens.append(p_tokens)
                    ps_vec.append(p_vec)
                    ps_len.append(p_len)
                ps_num = len(personas)

            dataset.append((us_id, us_tokens, us_vec, us_len, us_num, rs_tokens, rs_vec, rs_len, label, ps_tokens, ps_vec, ps_len, ps_num))
   
    return dataset


def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec


def charVec(tokens, charVocab, maxlen, maxWordLength):
    '''
    chars = np.array( (maxlen, maxWordLength) )    0 if not found in charVocab or None
    word_lengths = np.array( maxlen )              1 if None
    '''
    n = len(tokens)
    if n > maxlen:
        n = maxlen

    chars =  np.zeros((maxlen, maxWordLength), dtype=np.int32)
    word_lengths = np.ones(maxlen, dtype=np.int32)
    for i in range(n):
        token = tokens[i][:maxWordLength]
        word_lengths[i] = len(token)
        row = chars[i]
        for idx, ch in enumerate(token):
            if ch in charVocab:
                row[idx] = charVocab[ch]

    return chars, word_lengths


def batch_iter(data, batch_size, num_epochs, max_utter_num, max_utter_len, max_response_num, max_response_len, 
                max_persona_num, max_persona_len, charVocab, max_word_length, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            x_utterances = []
            x_utterances_len = []
            x_responses = []
            x_responses_len = []

            x_labels = []
            x_ids = []
            x_utterances_num = []

            x_utterances_char=[]
            x_utterances_char_len=[]
            x_responses_char=[]
            x_responses_char_len=[]

            x_personas = []
            x_personas_len = []
            x_personas_char=[]
            x_personas_char_len=[]
            x_personas_num = []

            for rowIdx in range(start_index, end_index):
                us_id, us_tokens, us_vec, us_len, us_num, rs_tokens, rs_vec, rs_len, label, ps_tokens, ps_vec, ps_len, ps_num = data[rowIdx]

                # normalize us_vec and us_len
                new_utters_vec = np.zeros((max_utter_num, max_utter_len), dtype='int32')
                new_utters_len = np.zeros((max_utter_num, ), dtype='int32')
                for i in range(len(us_len)):
                    new_utter_vec = normalize_vec(us_vec[i], max_utter_len)
                    new_utters_vec[i] = new_utter_vec
                    new_utters_len[i] = us_len[i]
                x_utterances.append(new_utters_vec)
                x_utterances_len.append(new_utters_len)

                # normalize rs_vec and rs_len
                new_responses_vec = np.zeros((max_response_num, max_response_len), dtype='int32')
                new_responses_len = np.zeros((max_response_num, ), dtype='int32')
                for i in range(len(rs_len)):
                    new_response_vec = normalize_vec(rs_vec[i], max_response_len)
                    new_responses_vec[i] = new_response_vec
                    new_responses_len[i] = rs_len[i]
                x_responses.append(new_responses_vec)
                x_responses_len.append(new_responses_len)

                x_labels.append(label)
                x_ids.append(us_id)
                x_utterances_num.append(us_num)

                # normalize us_CharVec and us_CharLen
                uttersCharVec = np.zeros((max_utter_num, max_utter_len, max_word_length), dtype='int32')
                uttersCharLen = np.ones((max_utter_num, max_utter_len), dtype='int32')
                for i in range(len(us_len)):
                    utterCharVec, utterCharLen = charVec(us_tokens[i], charVocab, max_utter_len, max_word_length)
                    uttersCharVec[i] = utterCharVec
                    uttersCharLen[i] = utterCharLen
                x_utterances_char.append(uttersCharVec)
                x_utterances_char_len.append(uttersCharLen)

                # normalize rs_CharVec and rs_CharLen
                rsCharVec = np.zeros((max_response_num, max_response_len, max_word_length), dtype='int32')
                rsCharLen = np.ones((max_response_num, max_response_len), dtype='int32')
                for i in range(len(us_len)):
                    rCharVec, rCharLen = charVec(rs_tokens[i], charVocab, max_response_len, max_word_length)
                    rsCharVec[i] = rCharVec
                    rsCharLen[i] = rCharLen
                x_responses_char.append(rsCharVec)
                x_responses_char_len.append(rsCharLen)

                # normalize ps_vec and ps_len
                new_personas_vec = np.zeros((max_persona_num, max_persona_len), dtype='int32')
                new_personas_len = np.zeros((max_persona_num, ), dtype='int32')
                for i in range(len(ps_len)):
                    new_persona_vec = normalize_vec(ps_vec[i], max_persona_len)
                    new_personas_vec[i] = new_persona_vec
                    new_personas_len[i] = ps_len[i]
                x_personas.append(new_personas_vec)
                x_personas_len.append(new_personas_len)

                # normalize ps_CharVec and ps_CharLen
                psCharVec = np.zeros((max_persona_num, max_persona_len, max_word_length), dtype='int32')
                psCharLen = np.ones((max_persona_num, max_persona_len), dtype='int32')
                for i in range(len(ps_len)):
                    pCharVec, pCharLen = charVec(ps_tokens[i], charVocab, max_persona_len, max_word_length)
                    psCharVec[i] = pCharVec
                    psCharLen[i] = pCharLen
                x_personas_char.append(psCharVec)
                x_personas_char_len.append(psCharLen)

                x_personas_num.append(ps_num)

            yield np.array(x_utterances), np.array(x_utterances_len), np.array(x_responses), np.array(x_responses_len), \
                  np.array(x_utterances_num), np.array(x_labels), x_ids, \
                  np.array(x_utterances_char), np.array(x_utterances_char_len), np.array(x_responses_char), np.array(x_responses_char_len), \
                  np.array(x_personas), np.array(x_personas_len), np.array(x_personas_char), np.array(x_personas_char_len), np.array(x_personas_num)

