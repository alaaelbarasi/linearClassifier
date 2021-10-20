from pandas.core.arrays import categorical
from pandas.core.arrays.sparse import dtype
from pandas.core.tools import numeric
from pandas.io import feather_format
import numpy as np 
import  pandas as pd 
import tensorflow as tf
import os
#load data
dftrain=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval= pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train=dftrain.pop('survived')
y_eval=dfeval.pop('survived')
categorical_columns=['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
numeric_columns=['age','fare']
feather_columns=[]
for feature_name in categorical_columns:
    vocabulary=dftrain[feature_name].unique()
    feather_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))

for feature_name in numeric_columns:
    feather_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

#input function 
def make_input_fn(data_df,label_df,num_epochs=10, shuffle=True,batch_size=32):
    def input_function():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
           ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn= make_input_fn(dftrain,y_train)
eval_input_fn=make_input_fn(dfeval,y_eval,num_epochs=1,shuffle=False)
linear_est = tf.estimator.LinearClassifier(feature_columns=feather_columns)
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

os.system('cls' if os.name == 'nt' else 'clear')  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model