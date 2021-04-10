## Description
* This code performs intent classification (over 150 intent classes) and
  out of scope (OOS) query detection on the imbalanced data set
  https://github.com/clinc/oos-eval/blob/master/data/data_imbalanced.json
* A sentence BERT model is used to encode the sentence into feature
  vector and MLP classifier is trained to predict 150 intent classes.
  
  
*  To detect, out of scope queries (queries whose intent is not covered
   by training set), Bayesian approach of out of distribution detection
   is used. 
   * First uncertainty of intent classification is computed using
   Monte Carlo dropout technique. 
   * A query whose uncertainty value >
   threshold is considered as out of scope. 
   * The hyperparameter thresold
   is found by optimizing the fscore over a small oos train set.



## Running the code 
* Install all the packages listed in requirements.txt. 
* Download the data files from https://github.com/clinc/oos-eval in same
  location as the notebook files. The file
  oos-eval/data/data\_imbalanced.json should be present relative to the
  location of notebook files.

* Then run the code in following order:


    *  feature\_extract.ipynb: extracts the features and saves them. This
code requires pytorch.

    * train.ipynb: loads the features, trains a model and saves the
  model.This code requires tensorflow 
  
    * eval.ipynb: loads the model perform prediction/evaluation.This code
 requires tensorflow.
    

