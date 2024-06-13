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

下記の手順をwindowsで実施したいです。

pyenvの導入から教えてください。

#### 仮想環境の構築方法として試した手順(結果動かすことはできなかった。)
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

#### `conda`を使用する方法(環境構築はできた方法-Intel Macでの動作は確認した。)
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

#### 環境構築後機械学習`code`を動かす手順
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

#### `UBFC-Phys`に対する知見
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

- `DeepPhys`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml
```

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

#### `windows`の`desktop PC`での環境構築方法
- `Intel Mac`での動作確認は大体できたので、`windows`の`desktop PC`での環境構築実行方法を検証する。
- 処理が重めのものが多いので、`Intel Mac`だけだと処理に時間がかかりすぎると判断したため。

- パッケージ管理をできるだけ行いたいので、`winget`(`Windows 10`以降であれば自動で入っている。)を使用する。
- `python`の`install`
- `power shell`を管理者権限で起動。
```:powershell
python
```
- 入力すると`app store`が開き`python`を`install`できる。
- `python`が`install`できたかと`version`を確認する。
```:powershell
python --version
```
- `pip`の`upgrate`(なんかパーミッションエラー出た。`pip`は使用できるのでそのまま使用しても良い。)
```:powershell
python -m pip install --upgrade pip
```
- `pip`のバージョン確認。
```:powershell
pip --version
```
- `pyenv-win`の`install`
```:powershell
pip install pyenv-win --target $HOME\.pyenv
```
- 環境変数の設定
```:powershell
[System.Environment]::SetEnvironmentVariable('PYENV', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
[System.Environment]::SetEnvironmentVariable('PYENV_ROOT', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
[System.Environment]::SetEnvironmentVariable('PYENV_HOME', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
[System.Environment]::SetEnvironmentVariable('PATH', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('PATH', "User"), "User")
```
- 環境変数の変更を反映させるために、PowerShellを再起動する。
- `pyenv`の動作確認
```:powershell
pyenv --version
```
- 必要な`python version`(3.8系)を`pyenv`から`install`
```:powershell
pyenv install 3.8.10
pyenv global 3.8.10
```
- 反映されているか確認
```:powershell
python --version
```
- 仮想環境の作成
```:powershell
python -m venv python-3.8.10.venv.rPPG
```
- 仮想環境のアクティベート
  - `path`の記載方法の問題で、移動して直接`activate`する必要があるかも。
```:powershell
python-3.8.19.venv.rPPG\Scripts\activate
```
- 必要なライブラリの`install`
  - `gpu`搭載の`pc`想定のコマンド
```:powershell
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```
- `requirements.txt`からライブラリをインストール
```:powershell
pip install -r requirements.txt
```
- 以上の手順で環境構築は完了した。
- `python main.py`を実施した際(`UBFC-Phys dataset`)に、前処理をかけた際に、メモリ容量エラーで前処理が正常に完了しなかった。
- 前処理が完了していないので、`input`データがないということで、最終の`output`出力にまでは至っていない。

### 2024/06/07：作業者：岡田
- `UBFC-Phys dataset`の内容について調査・理解する。
- 対象として`s1`を用いる。
- バイタルの計測機器は[`E4`](https://www.empatica.com/research/e4/)を使用している。
- データの同期方法は、`E4`のトリガリングボタンを押し、ライトが光るのでそれで同期を行っている。
- ただし、`vid_s1_T1.avi`にはトリガリングの瞬間がなかった。
  - 他のデータも見たが`T1`系のタスクにはトリガリングの瞬間がなさそう、、、
- `E4`で計測したデータは
  - `BVP`:`sampling rate 64Hz`
  - `EDA`:`sampling rate 4Hz`
- `bvp_s1_T1.csv`の行数
  - `11520/64=180s`
- `eda_s1_T1.csv`の行数
  - `720/4=180s`
- 上記の情報より、バイタルデータと動画データは同じ長さのデータが準備されていることがわかった。
- 感覚、同期したデータを入れて、同期した範囲の`output`でないとだめな気がするが、前処理の構造がどうなっているか確認する。

- 調査してわかったこと。
  - 前処理の設定値として、の2つがある模様。(`yaml`に記載されている。)
    - `DiffNormalized`
    - `Standardized`
  - 現状わからないこととして、設定箇所として
    - `DATA_TYPE`
    - `LABEL_TYPE`
    - の2つがあるが、`DATA_TYPE`が`list`形式で2つ指定できたりするのがよくわからない。
  - 前処理後のデータは、`.npy`形式で書き出される。
    - ビデオ側の`.npy`のデータと思われるものの形状
      - `s1_T1_input0.npy`の場合以下の状態
    ```:python
    print(data.shape)
    (128, 72, 72, 3)
    ```
      - おそらくだが、128フレーム、72*72の画像、RGB情報の3つという形状と思われる。
      - `s1_T1`の場合、49ファイルあった。
      - つまり`128 * 49 = 6272`相当
      - `6272 / 35 = 179.2s`相当(元動画は`180s`)
    - ラベル側の`.npy`のデータと思われるものの形状
      - `s1_T1_label0.npy`
    ```:python
    print(data.shape)
    (128, )
    ```
      - つまり`128 * 49 = 6272`相当
      - `6272 / 35 = 179.2s`相当(元動画は`180s`)

    - 思考
      - 疑問として、構造的に同期したデータを`input`しているわけではないと思うので、そのまま入れて評価として使用していいのか疑問。
      - 出力後に同期を取る方法もあるとは思うが、リサンプリング等もするので、本当にいいのろうか、、、

    - 顔の検出機能も前処理として入っていそう。
    - `CROP_FACE: BACKEND: 'HC'`の設定項目にて、`face detector`の種類を設置できそう。
    - 2種類指定することができ、
      - `HC:cv2.CascadeClassifier`を使用
      - `RF:RetinaFace.detect_faces`を使用
    - 該当箇所は`dataset/data_loader/BaseLoader.py`の`def face_detection()`関数に記載されている。
    

    - 処理想定
      - 画像の`raw`から顔を検出して、検出した顔の画像範囲を指定ピクセル`72*72`にリサイズしているぽい・・・？
      - 検証のために以下の`code`を入れて出力を確かめてみる。
      - `BaseLoader.py`の`def face_detection()`に`print(frame.shape)`を入れて`faced detector`に入れる画像を確かめる。
      - 出力`(1024, 1024, 3)`であったので、`face detect`は元動画のフレームの大きさで実施していそう。
      - `===face_box_coor===[225  45 789 789]`の出力が、異なっていたので、`face detect`した異なる矩形範囲を使用していそう。

### 2024/06/10：作業者：岡田
- 処理想定の確認のため、リサイズ前とリサイズ後の`png`の書き出し処理を`BaseLoader.py`の420行目ほどに追加。
```:python
# 415行目あたり
cv2.imwrite(f"original_{str(i)}.png", frame)
# 431行目あたり
cv2.imwrite(f"resized_be_original_{str(i)}.png", frame)
cv2.imwrite(f"resized_{str(i)}.png", resized_frames[i])
```
- 上記の`code`を追加し前処理を回すと、`top dir`に`png`ファイルが書き出される。
- 以下に出力画像を貼り付ける。色味がおかしいのは`RGB`の順序を間違えて出力しているからである。ピクセルの大きさが問題なので、そのままにしている。
- 元画像
  - 1024 * 1024
  ![オリジナル画像](./assets/original_0.png)
- `face detect`検出画像
  - 830 * 834
  ![`face detect`検出画像](./assets/resized_be_original_0.png)
- リサイズ画像
  - 74 * 74
  ![リサイズ画像](./assets/resized_0.png)

- 他の`face detect`検出画像を確認したがサイズが異なる場合があったので、検出された異なる画像サイズのものを指定サイズ（ここでは72 * 72）にして`input`としているのは確実と考えられる。

- 前処理の挙動を正確に把握する。
- `s1`のデータを入れた状態で下記の出力を確認する。
```:python
$python main.py --config_file ./configs/infer_configs/PURE_UBFC-PHYS_DEEPPHYS_BASIC.yaml
```
- 出力された`./PreprocessedData/UBFC-PHYS_SizeW72_SizeH72_ClipLength210_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len35_Median_face_boxTrue/s1_T1_input0.npy`の形状確認。
- (210, 72, 72, 6)
- 30ファイルあるので`210 * 30 = 6300`
- 元動画は`35fps`なので`6300 / 35 = 180s`

- 出力された`./PreprocessedData/UBFC-PHYS_SizeW72_SizeH72_ClipLength210_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len35_Median_face_boxTrue/s1_T1_label0.npy`の形状確認。
- (210,)

- `config`の`CHUNK_LENGTH: 210`で1回のエポックで入力するテンソルの長さを指定している模様。

- 調査必要だが、モデルによって`DATA_TYPE: [ 'DiffNormalized','Standardized' ]`が決まっていそう。
- ここが`list`の場合、`s1_T1_input0.npy`の出力の次元数がそのデータ分増えていた。(最終の次元がRGBと思われるので、2つの場合は6次元になっていた。)
- ラベルデータのスケーリングの指定は`LABEL_TYPE: DiffNormalized`で行う。
- 上記の処理の該当箇所は`BaseLoader.py`の`def preprocess(self, frames, bvps, config_preprocess):`関数あたり。


- 出力挙動の確認。
- `EXCLUSION_LIST`(除外)の`TASK_LIST`の指定と出力を確認する。
- 該当箇所は`UBFCPHYSLoader.py`の`def load_preprocessed_data(self):`関数箇所。

### 2024/06/11：作業者：岡田
- 出力`pickle`の中身を確認したところ、行数が5400行しかなかった。154s分相当である。
- 180sの動画なので、26sのデータがどこに行ったのか調査する。
- 色々と試行錯誤してみたが、詳細理由は現状ではわからなかった。
- 得られた知見として、
  - `yaml`の`CHUNK_LENGTH: 180`は前処理の際にまとめて書き出すフレーム数である。`.npy`ファイルのこと。
  - つまり、`35fps`の動画に対して`CHUNK_LENGTH: 35`で処理をかけると、動画の秒数分の`.npy`ファイルができる。
  - `BATCH_SIZE: 4`は推論時に一度に処理する`.npy`ファイルの数(一度に処理するエポックの数)だと思われる。
  - 一応、動画の`fps`数を`CHUNK_LENGTH`に指定し、`BATCH_SIZE`を1にすれば、動画の長さ分の出力は得られそう。
  `CHUNK_LENGTH: 210`で実施すると、5400行の出力となり`154s`分のデータにしかならない、、、なぜだ、、、

- オリジナルデータセットを各種モデルにかけられるようにデータの追加方法を調査する。
- データセットを`dataset dir`に移動。
- 独自で作成した`dataset name`は`Olive-VPPG`とした。
- 新しいデータ追加用プログラムとして`OliveVPPGLoader.py`を追加した。
- `LLM`に出力させた大筋の手順は以下。
```
**新しいデータセットを追加する手順**

1. [dataset/](file:///Users/takuma/Desktop/02_Github/rPPG-Toolbox/README.md#339%2C40-339%2C40) ディレクトリに新しいデータセットのファイル(例えば `dataset/MyNewDataset.py`)を作成します。このファイルには、データセットの詳細な説明が含まれています。

2. このファイル内に、データセットの名前と説明を書いてください。例えば、`MyNewDataset = "This dataset contains information about..."` のようにします。

3. `main.py` の中で、このデータセットファイルをインポートします。例えば、`from dataset.MyNewDataset import MyNewDataset` のようにします。

4. `main.py` の中で、データセットの名前空間を作成します。例えば、`MyNewDataset = MyNewDataset()`のようにします。

5. 最後に、`main.py` の中で、データセットの説明文を書いてください。例えば、`MyNewDataset = MyNewDataset("This dataset contains information about...")`のようにします。

6. このデータセットの説明文は、プログラムの実行時に参照されます。例えば、`MyNewDataset = MyNewDataset("This dataset contains information about...") `のようにします。

7. プログラムの実行中に、このデータセットの説明文は、データセットの名前空間に格納されます。

8. プログラムの終了時に、この説明文は、データセットの名前空間から取り出されます。

9. この説明文は、データセットの名前空間の中に格納されている情報を表示するために使われます。

10. この説明文は、プログラムの実行中に参照されます。

**注意:** この説明文は、プログラムの実行中に頻繁に参照されます。したがって、この説明文を注意深く読んでおくことが重要です。

プログラムの実行中に、この説明文に含まれる情報は、データセットの名前空間に格納されます。このデータセットの名前空間は、プログラムの終了時に参照されます。したがって、この説明文を注意深く読んでおくことが重要です。
```
- `Loader.py`の大筋の中身。
```：python
from BaseLoader import BaseLoader

class OriginalLoader(BaseLoader):
    def get_raw_data(self, raw_data_path):
        # ここにデータセット特有の生データ取得ロジックを実装
        pass

    def split_raw_data(self, data_dirs, begin, end):
        # データセットを訓練、検証、テスト用に分割するロジックを実装
        pass

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        # データの前処理ロジックを実装
        pass

    @staticmethod
    def read_video(video_file):
        # ビデオファイル読み込みの実装
        pass

    @staticmethod
    def read_wave(bvp_file):
        # 生理信号ファイル読み込みの実装
        pass
```

- 実際の手順
- `dataset/data_loader/OliveVPPGLoader.py`を作成した。
- `dataset/data_loader/__init__.py`に`OliveVPPGLoader`を追加した。
- `configs/infer_configs/PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml`の作成。
- `main.py`に`OliveVPPGLoader`も読み込めるように`code`編集。
- 実行`code`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml
```

- 問題のメモ
  - `BaseLoader.py`にて、`frames`の引数が動画全体を配列にしているため、処理が重いように見える。
  - ここを改善した方が良さそう。

- 知見のメモ
- `BaseLoader.py`の`def diff_normalize_data(data)`より、`DATA_TYPE`が`DiffNormalized`の場合、出力される最後の配列の値が0になるため、動画の最後のフレーム(もしかしたらエポック？)の予測はできなさそうな気がする。(要確認)

- とりあえず、改修終わった気がするので、回してみる。
```:bash
$conda activate rppg-toolbox
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml
```
- 諸々改修して動かせた！！
- 指の`PPG`がないデータもあるので`EXCLUSION_LIST`が機能するかはテスト必要。
- `BaseLoader.py`の`def save_multi_process`の一部処理変えたので、他のデータセットでも動くかテスト必要。(前処理プロセスを実施する必要あり)

### 2024/06/12：作業者：岡田
- 各種モデル等が正常に動作するかの確認を行う。
- 迅速なテストのために、`001.mp4`を20s程度にした動画で検証を実施した。
- カラーで描き出せるように`BaseLoader.py`の`cv2.imwrite`の処理を修正した。
- `BACKEND: RF`にして回してみた(下記)。
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml
```
- 問題なく動くか確認。
- 動作を見る限り`RF`の方が処理が重い模様。
- 動作は正常の模様。
- 以下の検証を行う。
  - `Olive-VPPG`のデータセットにて、全ての機械学習モデルが動作するか確認する。(左記に伴い`yaml`の作成も必要)
  - `UBFC-Phys`のデータセットにて数理モデル`UNSUPERVISED`が動作するか確認する。
- `UBFC-Phys`の`s1`の10s動画を作成。
- `UBFC-PHYS_UNSUPERVISED.yaml`で動作するか試してみる。
```:bash
$python main.py --config_file ./configs/infer_configs/UBFC-PHYS_UNSUPERVISED.yaml
```
- 出力される画像を確認したが、`UNSUPERVISED`なモデルでも画像は矩形範囲(皮膚のみがROIではない)ものをリサイズしたものを使用していた。
- `output`の`pikle`が出力されない。なぜだ。

### 2024/06/13：作業者：岡田
- `EXCLUSION_LIST`が機能するかテストする。
- `Olive-VPPG`の`EXCLUSION_LIST`にて、`002`を追加し、`002`の処理が回らないことを確かめる。
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml
```
- `OliveVPPGLoader.py`の`def load_load_preprocessed_data(self):`に`EXCLUSION_LIST`に関連する処理があるが、ここが対応できていなさそう。
- 余裕がある時に改修する。
- 挙動的に前処理は自動で全てのデータにかかりそう。
- `EXCLUSION_LIST`は推論をどれにするかの設定値ぽい？(ただテストすると、`002`を入れた時に動かないので詳細調査必要)

- `Olive-VPPG`に対応した`yaml`の全パターンの作成と検証を行う。

- `DeepPhys`
- `PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml
```

- `EfficentPhys`
- `PURE_Olive-VPPG_EFFICIENTPHYS.yaml`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_EFFICIENTPHYS.yaml
```

- `PhysFormer`
- `PURE_Olive-VPPG_PHYSFORMER.yaml`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_PHYSFORMER.yaml
```

- `PhysNet`
- `CHUNK_LENGTH: 128`と`DYNAMIC_DETECTION_FREQUENCY : 30`が一定値でないと正常に動作しないかもしれない。
- `PURE_Olive-VPPG_PHYSNET.yaml`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_PHYSNET.yaml
```

- `TS-CAN`
- `PURE_Olive-VPPG_TSCAN.yaml`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_TSCAN.yaml
```

- `UNSUPERVISED`
- `PURE_Olive-VPPG_UNSUPERVISED.yaml`
```:bash
$python main.py --config_file ./configs/infer_configs/PURE_Olive-VPPG_UNSUPERVISED.yaml
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