#### baseline_msa_sep
##### Repo for 2018 convai2 competition : submission for September-2018
Team Name: Mohd Shadab Alam  
Team member: Mohd Shadab Alam  

Scoring steps:  
1. Clone and install the ParlAI module, please ignore if latest version is already cloned  
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI  
cd ~/ParlAI; python setup.py develop   

2. Download this repo:   
git clone https://github.com/codesrepo/baseline_msa_sep.git  
(please note that files named "convai2_self_seq2seq_model*" are teacked using git lfs) 


3. After downloading, please copy "baseline_msa_sep" folder to "projects/convai2/" folder (of  ParlAI module)

4. Please copy folder "msa_agent" (from "baseline_msa_sep") to "parlai/agents/" folder (of ParlAI module) 

5. Please evaluate this model for ppl , f1 score and hits@1, i hope to better my ppl score with this submission
(Validation ppl while training: 27.56 )
Following validation scores were obtained:
PPL  (eval_ppl.py)  = 34.12   
f1 score using (eval_f1.py)  = 0.1708  
hits@1 using (eval_hits.py) = 0.134  

6. To get validation scores, please run the scripts present in ParlAI/projects/convai2/baseline_msa_sep/seq2seq/ folder:

To run the script please type these commands (from within baseline_msa_sep/seq2seq/ directory):
For ppl:python eval_ppl.py -dt valid
For f1:python eval_f1.py -dt valid
For hits@1:python eval_hits.py -dt valid

#####Running interactive.py

1. Implement steps 1 to 4 as mentioned above
2. Got to "ParlAI/projects/convai2/baseline_msa_sep/seq2seq/" folder
3. python 

#####Running the wild evaluation script

1. Implement steps 1 to 4 as mentioned above
2. Got to "ParlAI/projects/convai2/baseline_msa_sep/seq2seq/" folder
3. python convai_bot.py --bot-id '89ba1274-c5aa-4821-8932-c3c760fe546e' -rbu 'https://2258.lnsigo.mipt.ru/bot'

#####Training steps:

1. Training the language embeddings
	You need to run the script "lang_model.py" for generating and saving the language embeddings.
	Please make sure you have completed below steps before running this script
	a. Change the file paths as per your local path
	b. Download the pre trained weights "fwd_wt103.h5" and "itos_wt103.pkl" from this location "http://files.fast.ai/models/wt103/" please refer (https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb) for more details
	c. Go through all the comments starting from "----" and make appropriate changes
	d. Run this code
	e. Note down the location of the saved embeddings: "/home/shadab/ParlAI/lang_model/emb_array.npy" and saved indexes as these are required in training step 1

2. Training seq2seq model step 1
        a.Change the embedding paths in the "/home/shadab/ParlAI/projects/convai2/baseline_msa_sep/msa_agent/seq2seq/seq2seq_v0.py" script                 
                PRE_LM_PATH = "/home/shadab/ParlAI/lang_model/emb_array.npy"
                PRE_PATH = "/home/shadab/ParlAI/lang_model/tmp/itos.pkl"
        b.Ensure that (in seq2seq_v0.py):
            self.model.decoder.lt.weight.requires_grad = True
            self.model.encoder.lt.weight.requires_grad = True
	python train.py --dict-include-valid False -bs 100 -bsrt True -vp 25 -esz 700
3. Training seq2seq model step 2
        a.Ensure that (in seq2seq_v0.py):
            self.model.decoder.lt.weight.requires_grad = False
            self.model.encoder.lt.weight.requires_grad = False
	b. Data preparation: Run the data preparation code and rename the new training file
	python train.py --dict-include-valid False -bs 105 -bsrt True -vp 25 -esz 700

Otherlinks:
https://www.linkedin.com/in/mohd-shadab-alam/
https://www.kaggle.com/outliar

Thanks and Regards,
Shadab
