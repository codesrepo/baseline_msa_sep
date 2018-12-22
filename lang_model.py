from fastai.text import *
import html
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

#PATH=Path('data/aclImdb/')
LM_PATH = '/home/shadab/ParlAI/lang_model/' #----language model related files are stored at this path, create a "tmp" folder in "LM_PATH"

def get_texts(path):
    texts = []
    texts=open(path,'r').readlines()
    return np.array(texts)
trn_texts = get_texts('/home/shadab/ParlAI/data/ConvAI2/train_self_original.txt') #----Change path to train_self_original location
val_texts = get_texts('/home/shadab/ParlAI/data/ConvAI2/valid_self_original.txt') #----Change path to valid_self_original location

print(len(trn_texts))
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))

trn_texts = trn_texts[trn_idx]


trn_texts,val_texts = sklearn.model_selection.train_test_split(np.concatenate([trn_texts,val_texts]), test_size=0.1)
col_names = ['text','labels']
df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

df_trn.to_csv(LM_PATH+'train.csv',  index=False)
df_val.to_csv(LM_PATH+'test.csv',  index=False)

chunksize=24000
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ').replace('|', ' . ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls=1):
    labels = df.iloc[:,1].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df['text'].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    if 1==1:#for i, r in enumerate(df):
        print(1)
        tok_, labels_ = get_texts(df, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


df_trn = pd.read_csv(LM_PATH+'train.csv', header=None)
df_val = pd.read_csv(LM_PATH+'test.csv', header=None)





tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)




#np.save(LM_PATH+'tmp/'+'tok_trn.npy', tok_trn)#----In case you want to save tokens, uncomment this
#np.save(LM_PATH+'tmp/'+'tok_val.npy', tok_val)#----In case you want to save tokens, uncomment this


freq = Counter(p for o in tok_trn for p in o)
freq.most_common(25)

max_vocab = 60000
min_freq = 1

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)
trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

np.save(LM_PATH+'tmp/'+'trn_ids.npy', trn_lm)
np.save(LM_PATH+'tmp/'+'val_ids.npy', val_lm)
pickle.dump(itos, open(LM_PATH+'tmp/'+'itos.pkl', 'wb'))#----Saved token index to be used later in training

#trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
#val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
#itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))

vs=len(itos)
vs,len(trn_lm)




em_sz,nh,nl = 400,1150,3

PRE_PATH = PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

wgts = torch.load('/home/shadab/Downloads/fwd_wt103.h5', map_location=lambda storage, loc: storage)#----Load downloaded weights
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)

with open('/home/shadab/Downloads/itos_wt103.pkl', 'rb') as f:#----Load downloaded tokens
	itos2 = pickle.load(f)
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})


new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m


wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))


wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(LM_PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7


learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)



learner.model.load_state_dict(wgts)


lr=1e-3
lrs = lr

learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)


learner.save(LM_PATH+'lm_last_ft')

learner.load(LM_PATH+'lm_last_ft')

learner.unfreeze()

learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)

#learner.sched.plot()

learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)


learner.save(LM_PATH+'lm1')

learner.save_encoder(LM_PATH+'lm1_enc')

x=learner.model[0].encoder.data.cpu().numpy()
np.save("/home/shadab/ParlAI/lang_model/emb_array.npy",x)#----Path to save language embeddings

learner.sched.plot_loss()













           
