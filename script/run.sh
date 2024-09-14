dataset=camus_4ch
backbone=ResNet34
predictor=STGCNPredictor
loss_fn=a
npts=46
topk=5
device=0
nohup python main.py --train --resume --loss_fn ${loss_fn} --device ${device} --num_kpts ${npts} --top_k ${topk} --sample_rate 1.0 --engine tem_seq \
--backbone_name ${backbone} \
--predictor_name ${predictor} \
--cfg_path lib/config/config_sgd.yaml \
--data_path /home/lihh/Projects/data/${dataset} \
--proj_path experiments/${dataset}/${backbone}_${predictor}_${npts}pts_${topk}top > nohup/${dataset}/${backbone}_${predictor}_${npts}pts_${topk}top.out 2>&1 &
