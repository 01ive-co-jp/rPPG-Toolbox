






#%%
import tracemalloc

def my_function():
    # メモリ使用量を計測したい処理
    tracemalloc.start()  # トレース開始
    a = [1] * (10**6)
    b = [2] * (2 * 10**7)
    snapshot = tracemalloc.take_snapshot()  # スナップショットを取得
    top_stats = snapshot.statistics('lineno')  # 行単位で集計

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

    tracemalloc.stop()  # トレース終了

if __name__ == "__main__":
    my_function()


import numpy as np

# .npy ファイルを読み込む
#data = np.load('./PreprocessedData/UBFC-PHYS_SizeW72_SizeH72_ClipLength210_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len35_Median_face_boxTrue/s1_T1_input29.npy')

data = np.load('./PreprocessedData/UBFC-PHYS_SizeW72_SizeH72_ClipLength210_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len35_Median_face_boxTrue/s1_T1_label0.npy')



#s1_T1_label0.npy
# データを表示
print(data)


#%%
print(data.shape)


# %%
#48ファイルある

(128 * 48)/35
# %%


import os

def get_last_separator_index(path):
    """
    Returns the index of the last file separator in the given path.

    Args:
        path (str): The file path.

    Returns:
        int: The index of the last file separator.
    """
    return path.rfind(os.sep)

# Example usage
path = "/path/to/video/vid_s1_T1.avi"
last_separator_index = get_last_separator_index(path)
print(last_separator_index)  # Output: 13
# %%





#%%


import cv2

def trim_video(input_path, output_path, start_sec, end_sec):
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(input_path)
    
    # 動画のプロパティを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # トリミングの開始フレームと終了フレームを計算
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # 出力動画ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 動画をフレームごとに読み込み、指定範囲のフレームを出力動画に書き込む
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if start_frame <= current_frame < end_frame:
            out.write(frame)
        
        current_frame += 1
        if current_frame >= end_frame:
            break
    
    # リソースを解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 使用例
input_video_path = 'vid_s1_T1.avi'
output_video_path = 'vid_s1_T1_trim.avi'
start_seconds = 0  # トリミング開始秒数
end_seconds = 10    # トリミング終了秒数

trim_video(input_video_path, output_video_path, start_seconds, end_seconds)
# %%
