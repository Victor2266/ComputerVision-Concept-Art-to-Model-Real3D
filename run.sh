CUDA_VISIBLE_DEVICES=0 python run.py './assets/examples/GO_Bus.jpg' --output-dir output_demo --render --render-num-views 72 --model-save-format glb --mc-resolution 256 --pretrained-model-name-or-path './checkpoint/model_both_trained_v1.ckpt' #--no-remove-bg 

#if you get permission error then run chmod +x run.sh
