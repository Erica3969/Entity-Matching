import argparse
import jsonlines
from sklearn.metrics import *
import numpy as np

def evaluation(target_fn, prediction_fn):
    target=[]
    IDs=[]
    with jsonlines.open(target_fn,'r') as fh:
        for item in fh:
            IDs.append((item['ID1'], item['ID2']))
            target.append(int(item['label']))
    pred=[]
    sequences=[]
    with jsonlines.open(prediction_fn,'r') as fh:
        for item in fh:
            pred.append(int(item['match']))
            sequences.append((item['left'], item['right'], item["match_confidence"]))

    acc=accuracy_score(target,pred)
    f1=f1_score(target,pred)
    precision=precision_score(target,pred)
    recall=recall_score(target,pred)
    print(f" accuracy:\t{acc}\n precision:\t{precision}\n recall:\t{recall}\n f1:\t\t{f1}")
    TP=[]; TN=[]; FP=[]; FN=[]
    for i, label in enumerate(target):
        if label==pred[i]:
            if label==1: TP.append((IDs[i][0], IDs[i][1], sequences[i][0], sequences[i][1], sequences[i][2]))
            if label==0: TN.append((IDs[i][0], IDs[i][1], sequences[i][0], sequences[i][1], sequences[i][2]))
        if label!=pred[i]:
            if label==1: FN.append((IDs[i][0], IDs[i][1], sequences[i][0], sequences[i][1], sequences[i][2]))
            if label==0: FP.append((IDs[i][0], IDs[i][1], sequences[i][0], sequences[i][1], sequences[i][2]))
    TP = sorted(TP, key=lambda x:x[4],, reverse=True)
    TN = sorted(TN, key=lambda x:x[4],, reverse=True)
    FP = sorted(FP, key=lambda x:x[4],, reverse=True)
    FN = sorted(FN, key=lambda x:x[4],, reverse=True)
    np.savetxt(f'{prediction_fn}_TP.csv',np.asarray(TP),fmt="%s", delimiter='\t')
    np.savetxt(f'{prediction_fn}_TN.csv',np.asarray(TN),fmt="%s", delimiter='\t')
    np.savetxt(f'{prediction_fn}_FP.csv',np.asarray(FP),fmt="%s", delimiter='\t')
    np.savetxt(f'{prediction_fn}_FN.csv',np.asarray(FN),fmt="%s", delimiter='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected_output", "-e",type=str, default='data/amazon/relaxdays/all/test.jsonl')
    parser.add_argument("--prediction", "-p", type=str, default='output/amazon_all_ouput.jsonl')
    hp = parser.parse_args()

    # run evaluation
    evaluation(hp.expected_output, hp.prediction)
