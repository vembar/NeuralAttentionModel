import _dynet as dy
dyparams = dy.DynetParams()
dyparams.set_mem(11000)
dyparams.init()

import numpy as np
import random
import sys


np.random.seed(0)

batch_size = 32
input_size = 500
hidden_size = 500
attention_size = 200
num_layers = 1

german_train = 'train.en-de.low.filt.de'
english_train = 'train.en-de.low.filt.en'

german_valid = 'valid.en-de.low.de'
english_valid = 'valid.en-de.low.en'

german_test = 'test.en-de.low.de'
english_test = 'test.en-de.low.en'

def read_data(filename):
    sentences = []
    with open(filename) as f:
        for line in f:
            if line.strip() != "":
                tokens = line.strip().split(' ')
                sentences.append(['<s>'] + tokens + ['</s>'])
    return sentences

def get_vocab(sentences, limit):
    word_counts = {}
    for sample in sentences:
        for token in sample:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

    sorted_count = sorted(word_counts.items(), key=lambda x: (-x[1],x[0]))#[:limit-1]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_count)}
    id_to_item[len(id_to_item)+1] = '<UNK>'
    item_to_id = {v: k for k, v in id_to_item.items()}
    return id_to_item, item_to_id

def randomize(X,Y):
    c = zip(X, Y)
    random.shuffle(c)
    X = [e[0] for e in c]
    Y = [e[1] for e in c]
    return X,Y

def to_batch(X,Y):
    batched_X = []
    batched_Y = []
    X, Y = randomize(X,Y)
    for i in range(int(np.ceil(len(X)/batch_size))):
        batched_X.append(X[i*batch_size:(i+1)*batch_size])
        batched_Y.append(X[i*batch_size:(i+1)*batch_size])
    return batched_X, batched_Y

train_german = read_data(german_train)
train_english = read_data(english_train)

test_german = read_data(german_test)
test_english = read_data(english_test)

valid_german = read_data(german_valid)
valid_english = read_data(english_valid)

g_id_to_item, g_item_to_id = get_vocab(train_german, german_vocab)
e_id_to_item, e_item_to_id = get_vocab(train_english, english_vocab)

german_vocab = len(g_id_to_item)
english_vocab = len(g_id_to_item)

model = dy.Model()

enc_fwd_lstm = dy.LSTMBuilder(num_layers, input_size, hidden_size, model)
enc_bwd_lstm = dy.LSTMBuilder(num_layers, input_size, hidden_size, model)

dec_lstm = dy.LSTMBuilder(num_layers, hidden_size*2 + input_size, hidden_size, model)

input_lookup = model.add_lookup_parameters((german_vocab, input_size))

attention_w1 = model.add_parameters( (attention_size, hidden_size*2))
attention_w2 = model.add_parameters( (attention_size, hidden_size*num_layers*2))
attention_v = model.add_parameters( (1, attention_size))

decoder_wy = model.add_parameters( (english_vocab , hidden_size + hidden_size * 2))
decoder_by = model.add_parameters( (english_vocab))

output_lookup = model.add_lookup_parameters((english_vocab, input_size))

def pad_batch(batch, EOS):
    max_len = len(sorted(batch, key=lambda x: len(x))[-1])
    padded_batch = []
    for x in batch:
        x = x + [EOS]*(max_len-len(x))
        padded_batch.append(x)
    return padded_batch

def get_ided(sentences, vocab):
    input_set = []
    for sample in sentences:
        iset = []
        for token in sample:
            if token in vocab:
                iset.append(vocab[token])
            else:
                iset.append(vocab['<UNK>'])
        input_set.append(iset)
    return input_set

def encode(enc_fwd_lstm, enc_bwd_lstm, input_sent):
    dy.renew_cg()
    batch_size = len(input_sent)

    fwd_state = enc_fwd_lstm.initial_state()
    bwd_state = enc_bwd_lstm.initial_state()

    src_sentences = [dy.lookup_batch(input_lookup, wids) for wids in input_sent]
    src_sentences_rev = src_sentences[::-1]

    fwd_vectors = fwd_state.transduce(src_sentences)
    bwd_vectors = bwd_state.transduce(src_sentences_rev)[::-1]

    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
    return vectors

def attend(input_mat, seqlen, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    w2dt = w2*dy.concatenate(list(state.s()))
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    unnormalized = dy.reshape(unnormalized, (seqlen,), batch_size)
    att_weights = dy.softmax(unnormalized)
    
    context = input_mat * att_weights

    return context, att_weights

def decode(dec_lstm, vectors, output, out_mask):

    w = dy.parameter(decoder_wy)
    b = dy.parameter(decoder_by)
    w1 = dy.parameter(attention_w1)

    seqlen = len(vectors)

    input_mat = dy.concatenate_cols(vectors)

    w1dt = w1 * input_mat

    s = dec_lstm.initial_state()
    c_t_previous = dy.vecInput(hidden_size*2)

    loss = []
    assert len(output) == len(out_mask)
    for i in range(1,len(output)):
        last_output_embeddings = dy.lookup_batch(output_lookup, output[i - 1])
        vector = dy.concatenate([last_output_embeddings, c_t_previous])
        s = s.add_input(vector)
        h_t = s.output()
        c_t, alpha_t = attend(input_mat, seqlen, s, w1dt)

        h_c_concat = dy.concatenate([h_t, c_t])
        out_vector = dy.affine_transform([b, w, h_c_concat])

        loss_current = dy.pickneglogsoftmax_batch(out_vector, output[i])

        if 0 in out_mask[i]:
            mask_vector = dy.inputVector(out_mask[i])
            mask = dy.reshape(mask_vector, (1, ), batch_size)
            loss_current = loss_current * mask

        loss.append(loss_current)
        c_t_previous = c_t

    loss = dy.esum(loss)
    return dy.sum_batches(loss)/batch_size

def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors

def attention(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    w2dt = w2*dy.concatenate(list(state.s()))

    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)

    context = input_mat * att_weights
    return context

def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    sentence = [input_lookup[word] for word in in_seq]
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)

    bwd_vectors = list(reversed(bwd_vectors))
    encoded = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    w = dy.parameter(decoder_wy)
    b = dy.parameter(decoder_by)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[e_item_to_id['</s>']]
    c_t_prev = dy.vecInput(hidden_size * 2)
    s = dec_lstm.initial_state().add_input(dy.concatenate([c_t_prev, last_output_embeddings]))
    w1dt = w1dt or w1 * input_mat
    out = []
    count_EOS = 0
    for i in range(len(in_seq)+5):
        if count_EOS == 2: break
        vector = dy.concatenate([last_output_embeddings, c_t_prev])
        s = s.add_input(vector)
        h_t = s.output()
        c_t = attention(input_mat, s, w1dt)
        h_c_concat = dy.concatenate([h_t, c_t])
        out_vector = dy.affine_transform([b, w, h_c_concat])
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        c_t_prev = c_t
        if e_id_to_item[next_char] == '</s>':
            count_EOS += 1
            continue
        if next_char not in e_id_to_item:
            out.append('<UNK>')
        else:
            out.append(e_id_to_item[next_char])
    return ' '.join(out)

def get_loss(src, tgt, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, mask):
    encoded = encode(enc_fwd_lstm, enc_bwd_lstm, src)
    return decode(dec_lstm, encoded, tgt, mask)

train_german = get_ided(train_german, g_item_to_id)
train_english = get_ided(train_english, e_item_to_id)

valid_german = get_ided(valid_german, g_item_to_id)
valid_english = get_ided(valid_english, e_item_to_id)

test_german = get_ided(test_german, g_item_to_id)

def train():
    trainer = dy.SimpleSGDTrainer(model)
    for epoch in range(25):
        b_german, b_english = to_batch(train_german,train_english)
        for i in range(len(b_german)):
            bp_german = pad_batch(b_german[i], g_item_to_id['</s>'])
            bp_english = pad_batch(b_english[i], e_item_to_id['</s>'])

            gwids = []
            ewids = []
            masks = []
            for k in range(len(bp_german[0])):
                gwids.append([sent[k] if len(sent)>k else g_item_to_id['</s>'] for sent in bp_german])
            for k in range(len(bp_english[0])):
                ewids.append([sent[k] if len(sent)>k else e_item_to_id['</s>'] for sent in bp_english])
                masks.append([(1 if len(sent)> k else 0) for sent in b_english[i]])

            loss = get_loss(gwids, ewids, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, masks)
            loss_value = loss.value()
            loss.backward()
            trainer.update()

            ppl = np.exp(loss_value * batch_size / sum(len(s) for s in ewids))
            if i%10 == 0:
                print 'epoch %d, batch %d, loss=%f, ppl=%f' % (epoch, i, loss_value, ppl)
            if i%1000 == 0:
                vb_german, vb_english = to_batch(valid_german,valid_english)
                for m in range(len(vb_german)):
                    vbp_german = pad_batch(vb_german[m], g_item_to_id['</s>'])
                    vbp_english = pad_batch(vb_english[m], e_item_to_id['</s>'])
                    vgwids = []
                    vewids = []
                    vmasks = []
                    for k in range(len(vbp_german[0])):
                        vgwids.append([sent[k] if len(sent)>k else g_item_to_id['</s>'] for sent in vbp_german])
                    for k in range(len(vbp_english[0])):
                        vewids.append([sent[k] if len(sent)>k else e_item_to_id['</s>'] for sent in vbp_english])
                        vmasks.append([(1 if len(sent)> k else 0) for sent in vb_english[i]])
                    vloss = get_loss(vgwids, vewids, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, vmasks)
                    vloss_value = vloss.value()
                    vppl = np.exp(vloss_value * batch_size / sum(len(s) for s in vewids))
                    print 'Validation %d, batch %d, loss=%f, ppl=%f' % (epoch, m, vloss_value, vppl)
        model.save('single/saved_model'+str(epoch),[enc_fwd_lstm,enc_bwd_lstm,dec_lstm,input_lookup,attention_w1,attention_w2,attention_v,
                                     decoder_wy,decoder_by,output_lookup])

        print "Generating sentences"
        for sample in test_german:
            print generate(sample, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)

train()