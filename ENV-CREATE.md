## このドキュメントについて
- `rPPG-toolbox`の環境構築方法や作業記録などを記載したドキュメント。

## 作業記録
### 2024/06/05：作業者：岡田
- `rPPG-toolbox`の作業
    - 一通り、`rPPG-toolbox`を読んだ感じ、何はともあれデータセットが必要そうだった。
    - ただ、チュートリアルで指定されていたデータセットがフリーで`DL`できるデータセットではなかった。
    - 一旦フリーで`DL`ができるデータセットで機械学習系のモデルを動かす方法を模索する。
    - 機械学習系のモデルに関しては、既に学習済みのパラメータがあり、それを用いて、別のデータセットを評価するようなものが組まれていそう。
    - フリーで公開されていた
        - `SCAMPS`
        - `UBFC-Phys`
    - の2つのデータセットに対して、各モデルを実行する方法がないか調べる。
    - まずは`SCAMPS`のサンプルデータセットに対してモデルを動かす方法ないか調べる。
    - MODEL_PATH: "./final_model_release/SCAMPS_TSCAN.pth"は、SCAMPSデータセットを使用してTSCANモデルで学習した重みを指しており
    - データセット名:重み

- 調査してわかってきたことメモ
    - `./configs/infer_configs`の命名規則について
        - `./configs/infer_configs`には
            `${TRAIN_DATASET}_{TEST_DATASET}_{MODEL}_BASIC.yaml`
            の命名規則で、学習済みの機械学習モデルを動かすための`yaml`が入っている。
        - `./configs/train_configs`には
            `${TRAIN_DATASET}_${VAL_DATASET}_${TEST_DATASET}_${MODEL}_${VARIANT}.yaml`
        - 各部分の説明
        - `TRAIN_DATASET`: トレーニングに使用するデータセットの名前。
        - `VAL_DATASET`: バリデーションに使用するデータセットの名前。
        - `TEST_DATASET`: テストに使用するデータセットの名前。
        - `MODEL`: 使用するモデルの名前。
        - `VARIANT`: モデルや設定のバリアント（基本設定や特定のハイパーパラメータ設定など）。

    - `./final_model_release/`(学習済みモデルの重み)のの命名規則について記載する
        - 下記記載情報は明日確認すること。
        - `${TRAIN_DATASET}_{MODEL}.pth`

    - `UBFC-Phys`で動かす方法ないか考える(こちらは`TEST`があるので、1ファイルで動かす方法ないか設置値調整試してみる。)

- 上記の知見を踏まえて動かす手順として試してみるもの


- 使用するデータセット`UBFC-Phys`

## 仮想環境の構築方法として試した手順(結果動かすことはできなかった。)
    - `README.md`では`conda`を使用しているが仮想環境で動かせないか試す手順。
- 前提として`pyenv`が入っていること
```:bash
# 使用するバージョンを3.8.19にする
$pyenv global 3.8.19
# pythonのversion確認
$python --version
# 仮想環境の構築
$python -m venv python-3.8.19.venv
# 仮想環境のアクティベート
$source python-3.8.19.venv/bin/activate
```
- `rppg-toolbox`の`setup.sh`の内部に相当する処理
- 必要なライブラリなどを入れる。
```:bash
# PyTorch, torchvision, torchaudioのインストール
$pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
# 必要なlibraryをrequirements.txtからinstall
$pip install -r requirements.txt
```
- `m`チップ`mac`でエラー発生した。
- 対応として、`Rosetta`で動かしてみた。
- `Rosetta`でターミナルが開けているか確認するコマンド
- 出力が`x86_64`であればOK

```:bash
$uname -m
```

## `conda`を使用する方法(環境構築はできた方法-Intel Macでの動作は確認した。)
- 実行場所は、`rppg-toolbox`の`top dir`
- 前提として`homebrew`が入っていること。
- 環境を汚したくないので、`rppg-toolbox`とは少し異なる方法を採用している。
```:bash
# miniforgeのinstall
$brew install miniforge
# condaのバージョン確認(installできてるかの確認でもある)
$conda --version
# 自分の出力はconda 24.3.0だった。
```
- STEP 1: `bash setup.sh` の分解した内部コマンドを実施。

```:bash
# 関連づけられているcondaがある場合に備えて紐付けを解除するコマンド
$conda remove --name rppg-toolbox --all -y
# 必要なライブラリ等のinstall
# setup.shにはcudatoolkit=10.2の記載があるがGPUないので削除してコマンド使用
# gpuがないpc用コマンド
$conda create -n rppg-toolbox python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cpuonly -c pytorch -q -y
```
- 自動で`conda`起動されたくないので、手動アクティベート方法を採用(環境汚れるため)
```:bash
# condaのinstall先を探す
$conda info
# base environmentのpathを使用する
$source ${base environment output path}/bin/activate
```
- 上記のコマンドをまとめた(出力を使用して自動的にアクティベートするコマンド)
```:bash
$source $(conda info | grep 'base environment' | awk '{print $4}')/bin/activate
```
- `conda`のアクティベート
```:bash
$conda activate rppg-toolbox
```
- 必要ライブラリの`install`
```:bash
$pip install -r requirements.txt
```

- サンケイにある`Intel Mac`を使用した方法
- 上記記載の方法では、アーキテクチャエラーにより機械学習`code`を実施できなかった。
- サンケイの`Intel Mac`は`conda`が入っていたので若干手順が異なるので、以下に手順を記載する。
```:bash
# 構築環境の紐付け解除
$conda remove --name rppg-toolbox --all -y
# setup.shにはcudatoolkit=10.2の記載があるがGPUないので削除してコマンド使用
# gpuがないpc用コマンド
$conda create -n rppg-toolbox python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cpuonly -c pytorch -q -y
# 構築したconda環境のactivate
$conda activate rppg-toolbox
# 必要ライブラリのinstall
$pip install -r requirements.txt
```

## 環境構築後機械学習`code`を動かす手順
- `UBFC-Phys`の`s1`のデータがのみがある状態での作業を想定。
- `README.md`の`dataset`の節に記載されている方法でデータ配置してみる。
- `UBFC-Phys`の場合
- `rPPG-Toolbox`の`top-dir`で以下の`dataset`収納ディレクトリを作成する。
```:bash
# UBFC-Physのデータセットの置き場所を作成
$mkdir UBFC-PHYS/RawData
# 前処理後のデータが収納されるディレクトリを作成。
$mkdir PreprocessedData
```
- テストで使用する`yaml`として`PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml`を使用した。
- 変更箇所(変更後の値を記載)
    -  `METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']`
        - 算出される指標項目？`BA`が演算時に数式のエラーが発生したために除外した。どんな指標かどんな数式でエラーになっているかは追加で確認する必要がある。
    - `EXCLUSION_LIST: ['s1_T3', 's1_T2']`
        - テストデータとして使用する動画の`list`？`s1`の2つの動画をとりあえず指定してみた。`EXCLUSION_LIST`と`TASK_LIST`を1つにした場合にエラーが出た。2重の条件のためアルゴリズム的に全て処理を`pass`してしまうのが原因と考えている。`rPPG-Toolbox/dataset/data_loader/UBFCPHYSLoader.py", line 137, in load_preprocessed_dataraise ValueError(self.dataset_name + ' dataset loading data error!')`にてエラーが出た。
    - `TASK_LIST: ['T1', 'T2', 'T3']`
        - 対象するタスクの`list`?
    - `DATA_PATH: ${rPPG-ToolBoxまでのfull path}/UBFC-PHYS/RawData`
        - 使用するデータセットが格納されたディレクトリまでのフルパス。
    - `CACHED_PATH: ${rPPG-ToolBoxまでのfull path}/UBFC-PHYS/PreprocessedData`
        - 前処理後のデータを保存する予定のディレクトリまでのフルパス
    - `DO_PREPROCESS: true`
        - データの前処理をするかどうかの設定値。
        - 前処理をしていないデータがある時に1回だけ`ture`にしてデータの前処理を行うと良い。
        - 動画データの前処理だからか、処理時間が長いのと、なぜか`HDD`容量をかなり食って`PC`を落とすことがあることに注意。
        - 大体であるが、`15GB`の動画に対して`100GB`程度の`HDD`を一時的に消費したりしていた。
    - `DEVICE: cpu`
        - `gpu`がない環境で動作させる場合に変更が必要。
        - `cpu`で動かす場合、ここだけでなく、`python`の`code`の修正も必要だった。


- `python`の`code`で以下の点を修正。
- `main.py`の修正箇所
```:python
# 全ての関数記載前に以下の変数を追加
device = torch.device('cpu')

# model_trainer変数の代入箇所に、第3の変数としてdeviceを追加
model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict, device)
```
- `DeepPhysTrainer.py`の修正箇所
```:python
# 第4の引数として、deviceを追加
def __init__(self, config, data_loader, device=torch.device('cpu')):
# cpuへの紐付けcodeを実装
self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
```

- 実行
- 上記までの修正を行い、以下のコマンドを実施すると出力が得られた。
```:python
$python main.py --config_file ./configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml
```

- 出力場所と可視化について
- `runs`というディレクトリが作成され、その最下層に`PURE_DeepPhys_UBFC-PHYS_outputs.pickle`が作成されていた。
- 1つ上のディレクトリに以下2つのディレクトリがあった。`BA`を削除した影響で何か出ていないものがあるかもしれないのでそれは要調査。
    - `bland_altman_plots`
    - `saved_test_outputs`
- 出力された`pickle`を使用し、`/tools/output_signal_viz/data_out_viz.ipynb`の`pickle path`を変更することで、脈波?(要確認)の`plot`ができる


- エラーメッセージ記録
- `gpu`ないことによるエラー
```:bash
Traceback (most recent call last):
  File "main.py", line 311, in <module>
    test(config, data_loader_dict)
  File "main.py", line 97, in test
    model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
  File "/Users/olive_guest/Desktop/rPPG-Toolbox/neural_methods/trainer/DeepPhysTrainer.py", line 44, in __init__
    self.model = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/torch/nn/modules/module.py", line 927, in to
    return self._apply(convert)
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/torch/nn/modules/module.py", line 579, in _apply
    module._apply(fn)
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/torch/nn/modules/module.py", line 602, in _apply
    param_applied = fn(param)
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/torch/nn/modules/module.py", line 925, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/torch/cuda/__init__.py", line 211, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```
- `BA`あった時の計算エラー

```:bash
Traceback (most recent call last):
  File "main.py", line 313, in <module>
    test(config, data_loader_dict)
  File "main.py", line 106, in test
    model_trainer.test(data_loader_dict)
  File "/Users/olive_guest/Desktop/rPPG-Toolbox/neural_methods/trainer/DeepPhysTrainer.py", line 194, in test
    calculate_metrics(predictions, labels, self.config)
  File "/Users/olive_guest/Desktop/rPPG-Toolbox/evaluation/metrics.py", line 144, in calculate_metrics
    compare.scatter_plot(
  File "/Users/olive_guest/Desktop/rPPG-Toolbox/evaluation/BlandAltmanPy.py", line 106, in scatter_plot
    z = gaussian_kde(xy)(xy)
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/scipy/stats/kde.py", line 252, in evaluate
    result = gaussian_kernel_estimate[spec](self.dataset.T, self.weights[:, None],
  File "_stats.pyx", line 563, in scipy.stats._stats.gaussian_kernel_estimate
  File "<__array_function__ internals>", line 180, in cholesky
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/numpy/linalg/linalg.py", line 763, in cholesky
    r = gufunc(a, signature=signature, extobj=extobj)
  File "/Users/olive_guest/anaconda3/envs/rppg-toolbox/lib/python3.8/site-packages/numpy/linalg/linalg.py", line 91, in _raise_linalgerror_nonposdef
    raise LinAlgError("Matrix is not positive definite")
numpy.linalg.LinAlgError: Matrix is not positive definite
```
- 一旦上記記載の方法で動かすことができた。

## `UBFC-Phys`に対する知見
- 3つ目のタスクの動画が音声だけのものがあった。(`DL miss?`)


- やりたいこと
    - `M mac`で本当に動かないのか再度確認する。
    - 再度手順を試したが動かなかった。


### 2024/06/06：作業者：岡田
- 追加タスク
  - `DeepPhys`の挙動として、理解できないものの確認。
    - `EXCLUSION_LIST`と`TASKLIST`の組み合わせが想定通りに挙動しない問題。
    - `EXCLUSION_LIST`の要素が複数あるのに`pickle`ファイルが同数出力されない問題。
    - `METRICS`に`BA`がある場合に演算エラーが出る問題。
    - 出力データの構造の確認`plot`方法の確認
  - 他の機械学習モデルが動かす手順確認
- `UBFC-Phys`の学習済みのデータがあったモデルリスト
- モデルの形式上受け付けられない形式があるのか、全ての機械学習モデルの学習済みのデータセットがあったわけではなかった。
- `UBFC-Phys`の`yaml`があったモデルを記載する。
  - `DeepPhys`
  - `EfficentPhys`
  - `PhysFormer`
  - `PhysNet`
  - `TS-CAN`

- `EfficentPhys`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_UBFC-PHYS_EFFICIENTPHYS.yaml
```
- `PhysFormer`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_UBFC-PHYS_PHYSFORMER_BASIC.yaml
```
- `PhysNet`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_UBFC-PHYS_PHYSNET_BASIC.yaml
```
- `TS-CAN`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_UBFC-PHYS_TSCAN_BASIC.yaml
```






# rPPG-toolboxの実行できる仮想環境の作成手順書
- ベースの`python-version`の準備
```
pyenv install 3.8.19

pyenv global 3.8.19
```

- 仮想環境の作り方。

```
python -m venv python-3.8.19.venv
```

- アクティベート

```
source python-3.8.19.venv/bin/activate
```
- 必要なライブラリの`install`
```
pip3 install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=10.2 -c pytorch -q -y
```




conda remove --name rppg-toolbox --all -y
conda create -n rppg-toolbox python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=10.2 -c pytorch -q -y





3.8.19

- 仮想環境の作り方。
```
python -m venv python-3.11.9.venv
```

- 使用可能なversion一覧

```
pyenv install --list
```

```
pyenv install 3.11.9

pyenv global 3.11.9

```

pip3 install -r requirements.txt

アクティベート
source {フォルダ名}/bin/activate


source python-3.11.9.venv/bin/activate


https://github.com/ubicomplab/rPPG-Toolbox

上記のリポジトリの機械学習済みモデルを学習済みの重みを使用して実行したいです。
使用するデータセットは`UBFC-Phys`です。
ただし、保有している`UBFC-Phys`のデータセットは`s1`のみです。

この状態で
https://github.com/ubicomplab/rPPG-Toolbox/blob/main/configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml
を使用してDEEPPHYSを動かしたいです。

保有している`UBFC-Phys`の`s1`データセットの配置場所を教えてください。
また、実行できるように
https://github.com/ubicomplab/rPPG-Toolbox/blob/main/configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml
の内部で変更必要な箇所を教えてください。

また、その他実行に必要な情報があれば教えてください。


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>