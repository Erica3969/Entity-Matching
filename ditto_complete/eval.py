import argparse
import jsonlines
from sklearn.metrics import *

def evaluation(target_fn, prediction_fn):
    target=[]
    with jsonlines.open(target_fn,'r') as fh:
        for item in fh:
            target.append(int(item['label']))
    pred=[]
    sequences=[]
    with jsonlines.open(prediction_fn,'r') as fh:
        for item in fh:
            pred.append(int(item['match']))
            sequences.append((item['left'], item['right']))

    acc=accuracy_score(target,pred)
    f1=f1_score(target,pred)
    precision=precision_score(target,pred)
    recall=recall_score(target,pred)
    print(f" accuracy:\t{acc}\n precision:\t{precision}\n recall:\t{recall}\n f1:\t\t{f1}")
    TP=[]; TN=[]; FP=[]; FN=[]
    for i, label in enumerate(target):
        if label==pred[i]:
            if label==1: TP.append(sequences[i])
            if label==0: TN.append(sequences[i])
        if label!=pred[i]:
            if label==1: FN.append(sequences[i])
            if label==0: FP.append(sequences[i])
    np.savetxt(f'{prediction_fn}_TP.txt',np.asarray(TP),fmt="%s")
    np.savetxt(f'{prediction_fn}_TN.txt',np.asarray(TN),fmt="%s")
    np.savetxt(f'{prediction_fn}_FP.txt',np.asarray(FP),fmt="%s")
    np.savetxt(f'{prediction_fn}_FN.txt',np.asarray(FN),fmt="%s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected_output", "-e",type=str, default='data/amazon/relaxdays/all/test.jsonl')
    parser.add_argument("--prediction", "-p", type=str, default='output/amazon_all_ouput.jsonl')
    hp = parser.parse_args()

    # run evaluation
    evaluation(hp.expected_output, hp.prediction)
