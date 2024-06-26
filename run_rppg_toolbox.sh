#!/bin/bash

source $(conda info | grep 'base environment' | awk '{print $4}')/bin/activate
conda activate rppg-toolbox

declare -a yaml_files=(
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml"
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_EFFICIENTPHYS.yaml"
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_PHYSNET.yaml"
    #"./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_TSCAN.yaml"
    "./configs/infer_configs/Olive-VPPG/HC_PURE_Olive-VPPG_PHYSFORMER.yaml" #後から推論だけ回す必要あり
    #"./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_DEEPPHYS_BASIC.yaml"
    #"./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_EFFICIENTPHYS.yaml"
    #"./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_PHYSFORMER.yaml"
    #"./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_PHYSNET.yaml"
    #"./configs/infer_configs/Olive-VPPG/RF_PURE_Olive-VPPG_TSCAN.yaml"
)

for yaml_file in "${yaml_files[@]}"; do
    echo "${yaml_file} を実行します"
    start_time=$(date +%s)
    python main.py --config_file "${yaml_file}"
    exit_status=$?
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "${yaml_file} の実行時間: ${elapsed_time} 秒"

    if [ ${exit_status} -eq 0 ]; then
        echo "${yaml_file} が正常に終了しました"
    else
        echo "${yaml_file} がエラー終了しました"
    fi

    echo "" # 出力の区切り
done