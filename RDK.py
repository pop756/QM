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
