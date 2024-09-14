dataset=camus_4ch
backbone=ResNet34
predictor=STCPredictor
npts=46
topk=5
device=0
nohup python main.py --train --device ${device} --num_kpts ${npts} --top_k ${topk}  --sample_rate 1.0 --engine temporal \
