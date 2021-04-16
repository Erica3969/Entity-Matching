# Datensatzvorbereiten
- Produkte in 7 Kategorien: Arts_Crafts_and_Sewing, Home_and_Kitchen, Patio_Lawn_and_Garden, Pet_Supplie, Sports_and_Outdoor, Tool_and_Home_Improvement, Toys_and_Games  
- In jeder Kategorie sind jeweils 900 Produktpaare, davon 400 Matches und 500 Mismatches.  
- Ein Produktpaar wird als Match gelabled, wenn es in der Schnittmenge der similar_item und also_view und/oder also_buy liegt.  
- Ein Produktpaar wird als Mismatch gelabled, wenn es weder zusammenangeschaut noch zusammengekauft werden. Die Kategorien sind außerdem unterschiedlich.  
- Jedes Produkt hat 3 Attribute: title, feature und description.  
- Die Serializierung erfolgt wie im Paper beschrieben:  
```
COL title VAL ... COL feature VAL ... COL description VAL ... \t COL title VAL ... COL feature VAL ... COL description VAL ... \ label
```

# Modelltrainieren
## Eingabeparameter
```Bash
CUDA_VISIBLE_DEVICES=0 python train_ditto.py  
  --task amazon_relaxdays_cat_all  
  --batch_size 64  
  --max_len 128  
  --lr 3e-5  
  --n_epochs 20  
  --finetuning  
  --save_model  
  --logdir checkpoints/  
  --lm distilbert  
  --fp16  
  --da del  
  --dk product  
  --summarize  

```
## Ausgaben
```Bash
task_name: amazon_relaxdays_cat_all
=======================
step: 0, task: amazon_relaxdays_cat_all, loss: 0.01743592508137226
step: 10, task: amazon_relaxdays_cat_all, loss: 0.004206644371151924
step: 20, task: amazon_relaxdays_cat_all, loss: 0.001447625458240509
step: 30, task: amazon_relaxdays_cat_all, loss: 0.007100258022546768
step: 40, task: amazon_relaxdays_cat_all, loss: 0.0012025311589241028
step: 50, task: amazon_relaxdays_cat_all, loss: 0.00994708389043808
=========eval at epoch=10=========
Validation:
=============amazon_relaxdays_cat_all==================
accuracy=0.955
precision=0.953
recall=0.944
f1=0.949
======================================
Test:
=============amazon_relaxdays_cat_all==================
accuracy=0.945
precision=0.932
recall=0.950
f1=0.941
======================================

```