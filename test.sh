data=301_data
model=UNet
day=`date +"%m-%d"`
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=1
# nohup \
python -u test.py \
--data ${data} \
--log DiceCE_wce_${day} \
--model ${model} \
--output outputs/${model}_${day} \
--ckpt save_models_randomcrop/UNet_301_data_folder1/2025-04-03-200.pkl \
# > ${model}_${data}_DiceCE_wce_128_${day}.log 2>&1 &