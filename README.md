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


Otherlinks:
https://www.linkedin.com/in/mohd-shadab-alam/
https://www.kaggle.com/outliar

Thanks and Regards,
Shadab
