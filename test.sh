CUDA_VISIBLE_DEVICES=1 python3 main.py --arch="rtsrn" --test_model="CRNN" --batch_size=48  --STN  --sr_share --gradient  --use_distill --stu_iter=1 --vis_dir='vis/test' --mask --go_test --resume='ckpt/' --triple_clues --text_focus --vis

# resume just only needs to be filled in the folder such as 'ckpt/rstn-1/'
