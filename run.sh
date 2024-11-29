# If you get a permission error trying to run ./run.sh then run: chmod +x run.sh
#This Runs the Default Model
CUDA_VISIBLE_DEVICES=0 python run.py './assets/examples/apple.jpg' --output-dir output_demo --render --render-num-views 72 --model-save-format glb --mc-resolution 256 --pretrained-model-name-or-path './checkpoint/new_fruit_model_3000k.ckpt' #--no-remove-bg 


# This one will run our Sword Model
# CUDA_VISIBLE_DEVICES=0 python run.py './assets/examples/GO_Bus.jpg' --output-dir output_demo --render --render-num-views 72 --model-save-format glb --mc-resolution 256 --pretrained-model-name-or-path './checkpoint/sword_model_v1.ckpt' #--no-remove-bg 
