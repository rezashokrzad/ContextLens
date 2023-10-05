# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:02:18 2023

@author: Reza Shokrzad
"""

import base64
import pandas as pd
import io
import numpy as np
import os
import pickle

#embedding packages / BERT
import torch
# import transformers
from transformers import BertTokenizer, BertModel
# clustring
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
import scipy.stats
import itertools
#dimensionality reduction
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

#Embedding setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.eval()

#prepare DS for interactive visualization
def create_df5d(pca_bert_base_5C, method_name):
  inter_df = pd.DataFrame(pca_bert_base_5C)
  my_col = [method_name+'_Dim1', method_name+'_Dim2',
            method_name+'_Dim3', method_name+'_Dim4', method_name+'_Dim5',
            method_name+'_Dim6',method_name+'_Dim7',method_name+'_Dim8',
            method_name+'_Dim9',method_name+'_Dim10']
  inter_df.columns = my_col
  return inter_df

def emphasize_word(sent_list , word_label_string):
    def add_breaks(sentence):
        if len(sentence) <= 80:
            return sentence
        parts = []
        start = 0
        while start < len(sentence):
            parts.append(sentence[start:start + 80])
            start += 80
        return '<br>'.join(parts)
    
    for i in range(len(word_label_string)):
        sent_list[i] = sent_list[i].replace(word_label_string[i] , '<em>' + word_label_string[i] + '</em>')
        sent_list[i] = add_breaks(sent_list[i])

    return sent_list
def standardize_dataframe \
(sent_idx,sent_list, embs_umap_norm, embs_pca_norm , word_label , voted_label,weighted_vote_label,word_label_string,Label1,Label2):
    
    pca_bert_df = create_df5d(embs_pca_norm, method_name='PCA_BERT')
    umap_bert_df = create_df5d(embs_umap_norm, method_name='UMP_BERT')
    
    interactive_df = pd.concat([pca_bert_df, umap_bert_df], axis=1)
    interactive_df['text'] = emphasize_word(sent_list , word_label_string)
    interactive_df['Label1'] = np.zeros((len(sent_idx)),dtype=np.int16)
    if not Label1.empty:
        interactive_df['Label1'] = Label1.values[np.array(sent_idx)-1]
  
    interactive_df['Label2'] = word_label
    if not Label2.empty:
        interactive_df['Label2'] = Label2.values[np.array(sent_idx)-1]
    
    interactive_df['simple_majority_voting'] = voted_label
    interactive_df['weighted_majority_voting'] = weighted_vote_label
    interactive_df['idx'] = sent_idx
    interactive_df = interactive_df.reindex(columns=['idx', 'text','Label1', 'Label2',
                                                     'simple_majority_voting','weighted_majority_voting',
                                                     'PCA_BERT_Dim1', 'PCA_BERT_Dim2', 'PCA_BERT_Dim3', 'PCA_BERT_Dim4',
                                                     'PCA_BERT_Dim5','PCA_BERT_Dim6','PCA_BERT_Dim7',
                                                     'PCA_BERT_Dim8','PCA_BERT_Dim9','PCA_BERT_Dim10',
                                                     'UMP_BERT_Dim1', 'UMP_BERT_Dim2',
                                                     'UMP_BERT_Dim3', 'UMP_BERT_Dim4', 'UMP_BERT_Dim5',
                                                     'UMP_BERT_Dim6', 'UMP_BERT_Dim7', 'UMP_BERT_Dim8',
                                                     'UMP_BERT_Dim9', 'UMP_BERT_Dim10'
                                                     ])
    return interactive_df

def getEmbedingBERT(text, layer_num=8):
  marked_text = "[CLS] " + text + " [SEP]"

  # Split the sentence into tokens.
  tokenized_text = tokenizer.tokenize(marked_text)

  # Map the token strings to their vocabulary indeces.
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

  segments_ids = [1] * len(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  
  with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)

    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings[layer_num]
    
    # bert_semcor_index.append(idx)
  return token_embeddings


def get_bert_embs(doc, layer_num):
    
    print(str(doc.values[0][0]))
    print(str(doc.values[-1][0]))
    l1 = 10
    l2 = 10
    if len(doc.values[0][0]) < 10:
        l1 = len(doc.values[0][0])
    if len(doc.values[-1][0]) < 10:
        l2 = len(doc.values[-1][0])
        
    doc_save_name = './temp/'+doc.values[0][0][:l1] + doc.values[-1][0][-l2:]
    
    if os.path.isfile(doc_save_name):
        open_file = open(doc_save_name, "rb")
        em_list = pickle.load(open_file)
        open_file.close()
        
    else:
        em_list = []
        for _, d in doc.iterrows():
          text = d.values[0]
          try:
            em = getEmbedingBERT(text, layer_num)
            em_list.append(em)
          except:
            print('except ************')
            pass
        
        open_file = open(doc_save_name, "wb")
        pickle.dump(em_list, open_file)
        open_file.close()
    
    return em_list

def filter_embs(doc , embs, word,layer_num): # embs: list of sentences | word: list of words
    cnt = 0
    word_em_list = []
    word_label = []
    word_label_string = []
    sent_list = []
    sent_idx = []
    for i,sen in doc.iterrows():
        marked_text = "[CLS] " + sen.values[0] + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        lab = 0
        for w in word:
            try:
                idx = np.where(np.array(tokenized_text) == w)[0][0]
                word_em = embs[cnt][idx , :]
                word_em_list.append(word_em)
                word_label.append(lab)    
                word_label_string.append(w)
                sent_list.append(sen.values[0])
                sent_idx.append(i+1)
            except:
                tokenized_word = tokenizer.tokenize(w)
                if len(tokenized_word) > 1: # compound word
                    print(tokenized_word)
                    idx = -1
                    for ii in range(len(tokenized_text) - len(tokenized_word) + 1):
                        found = True
                        for j in range(len(tokenized_word)):
                            if tokenized_text[ii+j] != tokenized_word[j]:
                                found = False
                                break
                        if found:
                            print(ii)
                            idx = ii
                            break
                    if idx != -1: # found
                        temp = embs[cnt][idx:idx+len(tokenized_word) , :]
                        print(temp)
                        print(temp.shape)
                        word_em = torch.mean(embs[cnt][idx:idx+len(tokenized_word) , :],axis = 0)
                        print(word_em.shape)
                        print(word_em)
                        word_em_list.append(word_em)
                        word_label.append(lab)    
                        word_label_string.append(w)
                        sent_list.append(sen.values[0])
                        sent_idx.append(i+1)
                    
                else: # word not found in text
                    pass
                
            lab += 1      
        cnt += 1
    
    word_em_list = np.vstack(word_em_list)
    return word_em_list, word_label,word_label_string, sent_list, sent_idx


def normalization(emb_red):
    mm_sc = MinMaxScaler()
    emb_red_norm = mm_sc.fit_transform(emb_red)
    return emb_red_norm

#clustering setup
def clustering_kmeans(embs, n_cluster):
    clst_kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    embs_kmeans = clst_kmeans.fit_predict(embs)
    return embs_kmeans

def clustering_agg(embs, n_cluster):
    clst_agg = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
    embs_agg = clst_agg.fit_predict(embs)
    return embs_agg

#dimensionality reduction
def dimension_reduction_pca(embs):
    pca_5C = PCA(n_components=10, random_state=0)
    pca_embs = pca_5C.fit_transform(embs)
    return pca_embs

def dimension_reduction_umap(embs):
    umap_5C = UMAP(n_neighbors=10, n_components=10, random_state=42)
    umap_embs = umap_5C.fit_transform(embs)
    return umap_embs

#sense is the base. We compare all combinations of labels to find the best match.
def align_label(target_label, given_label):
  best_map = 0
  best_acc = 0
  for i in itertools.permutations(np.arange(len(set(given_label)))):
    map = np.array(i)
    mapped_labels = map[given_label]
    #accuracy coputaion
    if accuracy_score(target_label, mapped_labels) > best_acc:
      best_acc = accuracy_score(target_label, mapped_labels)
      best_map = map

  
  return best_map[given_label], best_acc

def label_alignment(gt, s1, s2, s3, s4, s5, s6):
    
    if sum(gt) == 0:
        gt = s1
        
    s1,_ = align_label(gt, s1)
    s2,_ = align_label(gt, s2)
    s3,_ = align_label(gt, s3)
    s4,_ = align_label(gt, s4)
    s5,_ = align_label(gt, s5)
    s6,_ = align_label(gt, s6)
    
    return s1,s2,s3,s4,s5,s6

def voting(s1, s2, s3, s4, s5, s6):
    return scipy.stats.mode((np.vstack((np.array(s1),np.array(s2),
            np.array(s3),np.array(s4),np.array(s5),np.array(s6)))))[0][0]

def weighted_majority_voting(gt , s1, s2, s3, s4, s5, s6, n_cluster):
    if sum(gt) == 0: 
        return voting(s1, s2, s3, s4, s5, s6)
    else:
        
        anns = np.concatenate((np.array(s1).reshape(-1,1),np.array(s2).reshape(-1,1),
                np.array(s3).reshape(-1,1),np.array(s4).reshape(-1,1),np.array(s5).reshape(-1,1),np.array(s6).reshape(-1,1)),axis = 1)
        
        annotator_expertise = np.zeros((6))
        _,annotator_expertise[0] = align_label(gt, s1)
        _,annotator_expertise[1] = align_label(gt, s2)
        _,annotator_expertise[2] = align_label(gt, s3)
        _,annotator_expertise[3] = align_label(gt, s4)
        _,annotator_expertise[4] = align_label(gt, s5)
        _,annotator_expertise[5] = align_label(gt, s6)
        
        weighted_mv = np.zeros((len(gt)))
        for row in range(len(gt)):
          weight_array = np.zeros((n_cluster))
          for ann in range(6):
            label = anns[row,ann]
            weight_array[label] += annotator_expertise[ann]
          weighted_mv[row] = np.argmax(weight_array)
          
        return weighted_mv
            

def get_dataframe(doc, word, Label1 , Label2, n_cluster=5):
    
    layer_num = 8
    embs = get_bert_embs(doc, layer_num)
    
    embs_filtered, word_label, word_label_string, sent_list, sent_idx = filter_embs(doc , embs, word, layer_num)
    
    cluster_kmeans_idx = clustering_kmeans(embs_filtered, n_cluster)
    cluster_agg_idx = clustering_agg(embs_filtered, n_cluster)
    embs_filtered = np.nan_to_num(embs_filtered, nan=0.0)

    embs_pca = dimension_reduction_pca(embs_filtered)
    embs_umap = dimension_reduction_umap(embs_filtered)
        
    cluster_pca_kmeans_idx = clustering_kmeans(embs_pca, n_cluster)
    cluster_umap_kmeans_idx = clustering_kmeans(embs_umap, n_cluster)
    cluster_pca_agg_idx = clustering_agg(embs_pca, n_cluster)
    cluster_umap_agg_idx = clustering_agg(embs_umap, n_cluster)
    
    [cluster_kmeans_idx, cluster_agg_idx, cluster_pca_kmeans_idx, cluster_umap_kmeans_idx, cluster_pca_agg_idx, cluster_umap_agg_idx] \
        = label_alignment(word_label, cluster_kmeans_idx,cluster_agg_idx,cluster_pca_kmeans_idx,cluster_umap_kmeans_idx,cluster_pca_agg_idx,cluster_umap_agg_idx)
    
    voted_label = voting(cluster_kmeans_idx,cluster_agg_idx,cluster_pca_kmeans_idx,cluster_umap_kmeans_idx,cluster_pca_agg_idx,cluster_umap_agg_idx)
    
    weighted_vote_label = weighted_majority_voting(word_label , cluster_kmeans_idx,cluster_agg_idx,cluster_pca_kmeans_idx,cluster_umap_kmeans_idx,cluster_pca_agg_idx,cluster_umap_agg_idx,n_cluster)
    
    embs_umap_norm = normalization(embs_umap)
    embs_pca_norm = normalization(embs_pca)
    
    df = standardize_dataframe \
    (sent_idx,sent_list, embs_umap_norm, embs_pca_norm , word_label ,voted_label,weighted_vote_label,word_label_string,Label1,Label2)
    
    df.to_excel('df.xlsx',index=False)
    print(df.columns)
    return df


def parse_contents(contents, filename, date):
    
    if contents == None:
        contents = 'No file read, try choosing a file'
        print("content is None")
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),header=None)
            #print(df.values)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            df = pd.DataFrame()
            return df
    except Exception as e:
        print(e)
        df = pd.DataFrame()
        return df

    print('df.values')
    print(np.shape(df.values))
    print(df.values[0][0])
    return df
    '''
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df_global.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_global.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
    '''
