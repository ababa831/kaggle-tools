# kaggle-tools　

Kaggleでよく使うツール置き場．本ドキュメントでは，ツールの使い方から新たに習得したテクニックまで，順次追加していく．

## EDA

## Feature engineering

## Machine learning
### LightGBM

特徴量の重要度をfeature importanceで可視化できるので（下図），他の手法でモデリングするときにも役に立つ．

![feature_importance](https://github.com/ababa893/kaggle-tools/blob/imgs/%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89.png?raw=true)

#### 準備

CPU版を使用する場合は`$ pip install lightgbm`．GPU版を使いたい場合は[ここ](http://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)を参考にする．

##### Datalabで使う場合
- 仮想環境のストレージ容量が気になる場合は，notebook上で以下のようにLGBMを永続ディスクにインストールし，ディレクトリのパスを指定しておく．
 
```
%bash
mkdir /dev/sdb/lgbm
pip install lightgbm -t /dev/sdb/lgbm
```

```
import sys
sys.path.append('/dev/sdb/lgbm/')
```

### LGBM.ipynb

LGBMのインストール～推論までの使用例をまとめたnotebook．Datalab上での使用を想定している．


Lightgbmによる学習と推論を実行を以下のコードで一纏めに行えるようになっている．
詳細は`LGBM.ipynb`にて．

```
DO(userows, train_df, test_df, sub_df, predictors, categoricals,  \
   debug=0, seed=7, fold_num=4, outs_path=<出力先のディレクトリ>)
```


### XGBoost

