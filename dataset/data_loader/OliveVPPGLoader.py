"""The dataloader for the Olive-VPPG dataset.

Details for the UBFC-PHYS Dataset see https://sites.google.com/view/ybenezeth/ubfc-phys.
If you use this dataset, please cite this paper:
R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, F. Yang. 
"UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress", 
IEEE Transactions on Affective Computing, 2021.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

from typing import Optional
import ast
import json

class OliveVPPGLoader(BaseLoader):
    """The data loader for the Olive-VPPG dataset."""

    def __init__(self, name, data_path, config_data):
        print("===UBFCPHYSLoader def init===")
        # あとから書き換えましょう
        """Initializes an UBFC-PHYS dataloader.
        
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- s1/
                     |       |-- vid_s1_T1.avi
                     |       |-- vid_s1_T2.avi
                     |       |-- vid_s1_T3.avi
                     |       |...
                     |       |-- bvp_s1_T1.csv
                     |       |-- bvp_s1_T2.csv
                     |       |-- bvp_s1_T3.csv
                     |   |-- s2/
                     |       |-- vid_s2_T1.avi
                     |       |-- vid_s2_T2.avi
                     |       |-- vid_s2_T3.avi
                     |       |...
                     |       |-- bvp_s2_T1.csv
                     |       |-- bvp_s2_T2.csv
                     |       |-- bvp_s2_T3.csv
                     |...
                     |   |-- sn/
                     |       |-- vid_sn_T1.avi
                     |       |-- vid_sn_T2.avi
                     |       |-- vid_sn_T3.avi
                     |       |...
                     |       |-- bvp_sn_T1.csv
                     |       |-- bvp_sn_T2.csv
                     |       |-- bvp_sn_T3.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        print("===name===data_path===config_data===")
        print(name,data_path,config_data)
        # 継承元クラスにname,data_path,config_dataを渡すことになるcode。
        super().__init__(name, data_path, config_data)

    # Oliveデータセットの場合
    # DATA_PATH: "/Users/olive_guest/Desktop/rPPG-Toolbox/UBFC-PHYS/RawData"
    # Oliveのデータセットの場合
    # DATA_PATH: "/Users/olive_guest/Desktop/rPPG-Toolbox/dataset/Olive-VPPG/dataset"
    # data_path = "/Users/olive_guest/Desktop/rPPG-Toolbox/dataset/Olive-VPPG/dataset"

    # raw-video
    # ここではデータセットのディレクトリを取得している。Olive-VPPGに対応する形に変更
    def get_raw_data(self, data_path):
        print("===get_raw_data===")

        """Returns data directories under the path(For Olive-VPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "raw-video" + os.sep + "*.mp4")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            '(.*).mp4', data_dir).group(1), "path": data_dir} for data_dir in data_dirs]

        print("===dirs===")
        print(dirs)

        return dirs

    # 使用するデータ量をbegin, endで指定できる箇所(変更不要)
    def split_raw_data(self, data_dirs, begin, end):
        print("===split_raw_data===")
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    # ここでは動画の読み込みとラベルの読み込みを行う。
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        print("===preprocess_dataset_subprocess===")
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # 動画の読み込み開始フレーム数を取得
        start_frame = self.get_skip_frame_num(saved_filename)

        # Read Frames
        # このファイル内部に処理がある静的メソッド
        # 読み込むvideoのpathを渡している。
        frames = self.read_video(
            video_file = os.path.join(data_dirs[i]['path']),
            start_frame = start_frame
            )

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            # Olive-VPPGでは、bvpデータはbiosignalsplux-dataに保存されているのでそれに対応
            bvps = self.read_wave(
                bvp_file = self.generate_bvp_file_path(saved_filename),
                saved_filename = saved_filename,
                # ecgでもできるようにしたいがyamlの設定値から引けるようにか何か将来的に工夫したい
                use_grand_trough = 'ppgFingertip'
                )
                #config_preprocessを使用することでワンチャンできそう。

        # Ground truthを動画のfps(つまり動画のフレームレート)に合わせる処理箇所
        # ここでresampleするので grand_troughのsampling_rateは気にしなくて良い(video fpsにされるため)
        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])
            
        # baseloaderにある
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        print("===load_preprocessed_data===")
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """

        print("===self.file_list_path===")
        print(self.file_list_path)

        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        # この辺の様子がおかしい
        for input in base_inputs:
            print("===input===")
            print(input)
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]

            if self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST :
                print("=self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST")
                # Skip loading the input as it's in the exclusion list
                continue
            # ここいらない気がする
            if self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST):
                print("=self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST")
                # Skip loading the input as it's not in the task list
                continue
            filtered_inputs.append(input)

        print("===filtered_inputs===")
        print(filtered_inputs)

        if not filtered_inputs:
            print("===self.dataset_name===")
            print(self.dataset_name)
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)


    def get_skip_frame_num(self, saved_filename : str):
        print("===get_skip_frame_num===")
        """Returns the number of frames to skip for the LED detection."""
        # annotationsの情報からskipすべきフレーム数を取得
        annotation_path = self.generate_annotation_file_path(saved_filename)

        # jsonをdictとして読み込む関数
        annotation_dict = self.load_json_as_dict(annotation_path)

        # LED の判定フレーム数を取得
        start_frame = int(annotation_dict['videoSyncFrame'])

        return start_frame

    # ここでは、bvpデータはbiosignalsplux-dataに保存されているのでそれに対応
    def read_wave(self, bvp_file: str, saved_filename: str, use_grand_trough: str = 'ppgFingertip'):
        print("===bvp_file===")
        """Reads a bvp signal file."""
        # annotationsの情報からskipすべきフレーム数を取得
        annotation_path = self.generate_annotation_file_path(saved_filename)

        # jsonをdictとして読み込む関数
        data_dict = self.load_json_as_dict(annotation_path)

        # ECGが上下判定している場合反転させる係数
        if data_dict['biosignalspluxEcgUpsideDown'] == "TRUE":
            bio_coe = -1
        elif data_dict['biosignalspluxEcgUpsideDown'] == "FALSE":
            bio_coe = 1
        # LEDが消灯した時のフレーム数を使用する場合1にする
        if data_dict['ledSyncConditions'] == "on":
            LED_off = 0
        elif data_dict['ledSyncConditions'] == "off":
            LED_off = 1

        # biopluxの読み込みで使用するカラム名list
        column_names = ast.literal_eval(data_dict['biosignalspluxCols'])
        
        # 追加しないと読み込みがおかしくなるので追加
        column_names.append('dummy-dummy')

        # biosignalspluxのデータの読み込み
        df_biosignalsplux = pd.read_csv(
            bvp_file,
            sep="\t",
            names = column_names,
            # ここは固定(biopluxの固定のheader行数なので)
            skiprows = 3
            )

        # 'LED_signal'列で最初に1という値が出てくる行のインデックスを取得（LEDの点滅箇所）
        # 場合によって消灯のポイントになることもあるので、条件分岐させておく
        if LED_off == 1:
            # 1の連続が始まる最初のインデックス
            start_index = df_biosignalsplux['ledSignal'].eq(1).idxmax()
            # start_index以降で、値が0になる最初のインデックス
            index = df_biosignalsplux['ledSignal'][start_index:].eq(0).idxmax()
        else:
            index = df_biosignalsplux['ledSignal'].eq(1).idxmax()

        print(f"index-number:{index}")
        # 該当の行より上の行のみを保持
        df_biosignalsplux = df_biosignalsplux.loc[index:].reset_index(drop=True)
        # 目視確認により岡田の波形は上下反転しているため「-1」をかけることにより反転
        df_biosignalsplux['ecg'] = df_biosignalsplux['ecg'] * bio_coe

        # grand_troughの値をlistに変換
        # use_grand_troughで指定されたものを使う
        bvp = df_biosignalsplux[use_grand_trough].astype(float).tolist()

        return np.asarray(bvp)

    # jsonをdictとして読み込む関数
    @staticmethod
    def load_json_as_dict(file_path):
        """Loads a json file as a dictionary."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data


    @staticmethod
    def generate_bvp_file_path(saved_filename : str):
        print("===generate_bvp_file_path===")
        """Returns the path of the bvp file."""
        # ディレクトリとファイル名を分離
        directory, filename_no_ext = os.path.split(saved_filename)

        # 新しいディレクトリパスを構築
        bvp_directory = os.path.join(os.path.dirname(directory), "biosignalsplux-data")

        # 新しいファイル名を構築
        bvp_filename = filename_no_ext + ".txt"

        # 新しい完全なパスを構築
        bvp_path = os.path.join(bvp_directory, bvp_filename)

        return bvp_path

    @staticmethod
    def generate_annotation_file_path(saved_filename : str):
        print("===generate_annotation_file_path===")
        """Returns the path of the annotation file."""
        # ディレクトリとファイル名を分離
        directory, filename_no_ext = os.path.split(saved_filename)

        # 新しいディレクトリパスを構築
        annotation_directory = os.path.join(os.path.dirname(directory), "annotations")

        # 新しいファイル名を構築
        annotation_filename = filename_no_ext + ".json"

        # 新しい完全なパスを構築
        annotation_path = os.path.join(annotation_directory, annotation_filename)

        return annotation_path

    # Olive-VPPGにおいては、動画の途中から読み込む処理しか存在しないため読み込み開始フレーム数の指定を必須にしている。
    @staticmethod
    def read_video(video_file: str, start_frame: Optional[int] = None):
        print("===read_video===")
        """Reads a video file from a specified frame to the end, returns frames(T,H,W,3)
        
        Args:
            video_file (str): Path to the video file.
            start_frame (Optional[int]): Index of the first frame to read from. Must not be None.
        """
        if start_frame is None:
            raise ValueError("start_frame must be specified and cannot be None")

        VidObj = cv2.VideoCapture(video_file)
        # Move to the start frame
        VidObj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        success, frame = VidObj.read()
        frames = list()

        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)