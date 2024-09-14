dataset=camus_4ch
backbone=ResNet34
predictor=STCPredictor
npts=46
topk=5
device=0
nohup python main.py --train --device ${device} --num_kpts ${npts} --top_k ${topk}  --sample_rate 1.0 --engine temporal \
--backbone_name ${backbone} \
--predictor_name ${predictor} \
--cfg_path lib/config/config_sgd.yaml \
--data_path Data/${dataset} \
--proj_path experiments/${dataset}/${backbone}_${predictor}_${npts}pts_${topk}top > nohup/${dataset}/${backbone}_${predictor}_${npts}pts_${topk}top.out 2>&1 &
