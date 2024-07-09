# windows用
# 実行環境に入ってからこのファイルを実行する。
import subprocess
import time
from pathlib import Path
import sys
import subprocess
from threading import Thread
from queue import Queue, Empty

print(sys.executable)

venv_python_path = r"C:\Users\Olive\Desktop\PPG-Toolbox\python-3.8.10.venv.rPPG\Scripts\python.exe"

# YAMLファイルのリストを定義
yaml_files = [
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml",
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_EFFICIENTPHYS.yaml",
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_PHYSNET.yaml",
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_TSCAN.yaml",
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_PHYSFORMER.yaml", 
    "./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml",
    "./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_EFFICIENTPHYS.yaml",
    "./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_PHYSFORMER.yaml",
    "./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_PHYSNET.yaml",
    "./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_TSCAN.yaml",
]

def stream_watcher(identifier, stream):
    for line in stream:
        line = line.decode()
        print(f"{identifier}: {line}", end="")

for yaml_file in yaml_files:
    yaml_file_path = Path(yaml_file)
    if yaml_file_path.exists():
        print(f"{yaml_file} を実行します")
        start_time = time.time()

        q = Queue()
        proc = subprocess.Popen([venv_python_path, "main.py", "--config_file", str(yaml_file_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)

        stdout_watcher = Thread(target=stream_watcher, args=("STDOUT", proc.stdout, q))
        stderr_watcher = Thread(target=stream_watcher, args=("STDERR", proc.stderr, q))
        stdout_watcher.start()
        stderr_watcher.start()

        while True:
            try:
                proc.wait(0.1)
                break
            except subprocess.TimeoutExpired:
                try:
                    line = q.get_nowait()
                except Empty:
                    pass

        stdout_watcher.join()
        stderr_watcher.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{yaml_file} の実行時間: {elapsed_time:.2f} 秒")

        if proc.returncode == 0:
            print(f"{yaml_file} が正常に終了しました")
        else:
            print(f"{yaml_file} がエラー終了しました (返り値: {proc.returncode})")

        print("")
    else:
        print(f"{yaml_file} が存在しません")

# python main.py --config_file ./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml
#  C:\Users\Olive\Desktop\PPG-Toolbox\python-3.8.10.venv.rPPG\Scripts\python.exe