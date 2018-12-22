import codecs
import ast

conv_dict={}
#download the corpus from here: http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
input_files=["/home/shadab/Downloads/cornell_movie_dialogs_corpus/movie_lines.txt"]
#Splitting responses and treating tab-space
for fname in input_files:
        with codecs.open(fname, "r",encoding='utf-8', errors='ignore') as infile:
            for line in infile:
                splitted_line = line.split("+++$+++")
                d_key = splitted_line[0].strip()
                d_val = splitted_line[-1][0:-2].replace("\t", " ")
                conv_dict[d_key] = d_val

all_utterences=[]
input_files=["/home/shadab/Downloads/cornell_movie_dialogs_corpus/movie_conversations.txt"]
for fname in input_files:
        with codecs.open(fname, "r",encoding='utf-8', errors='ignore') as infile:
            for line in infile:
                splitted_line = line.split("+++$+++")
                list_dialogue = splitted_line[-1].strip()
                all_utterences.append(ast.literal_eval(list_dialogue))
#Providing serial number to each row as in convai data
with open("/home/shadab/ParlAI/data/ConvAI2/cornell_dialogue.txt","w") as outfile:
    for l in all_utterences:
        c=0
        sno=0
        prev_line = " "
        for i in l:
                if c>20:
                    continue
                if len(conv_dict[i].split())>15:
                    conv_dict[i] = conv_dict[i][0:50]
                if c%2==0:
                    prev_line = str(sno) + " " + conv_dict[i]
                if c%2==1:
                    new_line = prev_line + "\t" + conv_dict[i] +"\n"
                    sno+=1
                    outfile.write(new_line)
                c+=1
outfile.close()
#Appending cornell data with parlai data
input_files=['/home/shadab/ParlAI/data/ConvAI2/train_self_original.txt','/home/shadab/ParlAI/data/ConvAI2/cornell_dialogue.txt']
with open("/home/shadab/ParlAI/data/ConvAI2/temp_cornell.txt","w") as outfile:
    for fname in input_files:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
outfile.close()
#Rename "temp_cornell.txt" to "train_self_original.txt"
















