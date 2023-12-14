from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import numpy as np
def smile_to_RDkit(smiles_string,bits = 1024):
    # 예시 SMILES 문자열

    # SMILES 문자열을 RDKit 분자 객체로 변환
    result = []
    for temp in smiles_string:
        molecule = Chem.MolFromSmiles(temp)
        if molecule is not None:  
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=bits)
            fingerprint_bits = list(fingerprint)
            result.append(fingerprint_bits)

        else:
            result.append([1]*bits)
    return result
class Drug:
    def __init__(self,data,target):
        Drug.x = data
        Drug.y = target

def Result(model,All_features,All_labels,SPLITS=4,plot=True):
    n_iter = 0
    kf = KFold(n_splits = SPLITS,shuffle=True)
    AUC_res = []
    accuracy_res = []
    precision_score_res = []
    recall_score_res = []
    f1_score_res = []
    for train_idx,test_idx in kf.split(All_features,All_labels):
        print(f'--------------------{n_iter} KFold-------------------')
        train_features,train_labels = All_features[train_idx],All_labels[train_idx]
        test_features,test_labels = All_features[test_idx],All_labels[test_idx]

        print(f'train_idx_len : {len(train_labels)} / test_idx_len : {len(test_labels)}')
        model.fit(train_features ,train_labels)
        train_accuracy = model.score(train_features ,train_labels)
        accuracy = model.score(test_features,test_labels)
        y_scores = model.predict_proba(test_features)[:, 1]  # 양성 클래스에 대한 예측 확률
        y_pred = model.predict(test_features)
        fpr, tpr, thresholds = roc_curve(test_labels,y_scores)
        roc_auc = auc(fpr, tpr)
        if plot:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'fold {n_iter} ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'fold {n_iter} ROC Curve Neural')
            plt.legend(loc='lower right')
            plt.show()
        res1 = precision_score(test_labels,y_pred)
        res2 = recall_score(test_labels,y_pred)
        res3 = f1_score(test_labels,y_pred)
        print(f'AUC: {roc_auc:.3f}')
        print(f"Test set accuracy: {accuracy:.3f}, Train set accuracy : {train_accuracy:.3f}")
        print(f"precision : {res1:.3f}")
        print(f"recall : {res2:.3f}")
        print(f"f1_score : {res3:.3f}")
        n_iter += 1
        AUC_res.append(roc_auc)
        accuracy_res.append(accuracy)
        precision_score_res.append(res1)
        recall_score_res.append(res2)
        f1_score_res.append(res3)
    print("\n")
    print("average values")
    print(f'AUC: {np.mean(AUC_res):.3f}')
    print(f"Test set accuracy: {np.mean(accuracy_res):.3f}")
    print(f"precision : {np.mean(precision_score_res):.3f}")
    print(f"recall : {np.mean(recall_score_res):.3f}")
    print(f"f1_score : {np.mean(f1_score_res):.3f}")
import codecs
from SmilesPE.tokenizer import *
from tqdm import tqdm
spe_vob= codecs.open('Practice/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
def smile_tokenize(smile_list):
    corpus = []
    for i in (smile_list):
        temp = spe.tokenize(i)
        temp = temp.strip().split(' ')
        corpus.append(temp)
    return corpus


from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
def embeding_model(corpus,vector_size=256, workers=4, sg=1, compute_loss=True, epochs=500,min_count=1,sv=(-1,None)):
    class callback(CallbackAny2Vec):
        """Callback to print loss after each epoch."""

        def __init__(self):
            self.epoch = 0
            self.loss_to_be_subed = 0

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            loss_now = loss - self.loss_to_be_subed
            self.loss_to_be_subed = loss
            print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
            self.epoch += 1
            if sv[0] !=-1 and self.epoch>sv[0]:
                model.save(sv[1])

    print("학습 중")
    model = Word2Vec(corpus, vector_size=vector_size, workers=workers, sg=sg, compute_loss=compute_loss, epochs=epochs,min_count=min_count, callbacks=[callback()])
    print('완료')
    return model

def string_to_vec(model,corpus,max_):
    res = []
    for line in corpus:
        temp = []
        for word in line:
            try:
                temp.append(model.wv[word])
            except:
                temp.append(np.array([0]*model.vector_size))
        while(len(temp)<max_):
            temp.append(np.array([0]*model.vector_size))
        res.append(temp)
    return res

import selfies as sf

def SMILE_to_SELFIES_TOKEN(train_set):
    res = []
    # SMILES -> SELFIES -> SMILES translation
    for index,i in enumerate(train_set):
        benzene_sf = sf.encoder(i)  # [C][=C][C][=C][C][=C][Ring1][=Branch1]
        if index%10000 == 0:
            print(index)
        res.append(list(sf.split_selfies(benzene_sf)))
    return res
    # ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[=Branch1]']