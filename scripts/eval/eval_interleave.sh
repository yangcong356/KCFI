alias python=python3
CKPT_PATH=$1

# 获取路径中 / 的数量
num_slashes=$(echo "$CKPT_PATH" | awk -F'/' '{print NF-1}')

if [ "$num_slashes" -gt 6 ]; then
    # 如果路径中的 / 超过一定数量（这里以10为例），提取最后两个部分
    NAME=$(echo "$CKPT_PATH" | awk -F'/' '{print $(NF-1)"/"$NF}')
else
    # 否则提取最后一个部分
    NAME=$(echo "$CKPT_PATH" | awk -F'/' '{print $NF}')
fi

echo $NAME
##### set images path 
DATA_PATH=$3/images
JSON_PATH=$3/ann-llava/$4.json
MODEL_BASE=$2
############################### eval multi-image 
RESULT_NAME="/your path/KCFI/results/${NAME}"
echo $RESULT_NAME

mkdir -p results/${NAME}

file_path=${RESULT_NAME}/result.jsonl

bash scripts/eval/eval_multiprocess.sh \
${CKPT_PATH} \
${MODEL_BASE} \
${JSON_PATH} \
${RESULT_NAME} \
${DATA_PATH} \
"" \
2 \
0.2

# python3 llava/eval/evaluate_interleave.py --result-dir ${RESULT_NAME}
