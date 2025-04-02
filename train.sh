data=301_data
model=DSCNet
day=`date +"%m-%d"`
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=3
# nohup \
python -u train.py \
--data ${data} \
--log DiceCE_wce_${day} \
--model ${model} \
# > ${model}_${data}_DiceCE_wce_192_${day}.log 2>&1 &