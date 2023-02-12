CUDA_VISIBLE_DEVICES=0 python3 main.py --arch="rtsrn" --test_model="CRNN" --batch_size=48 --STN  --sr_share --gradient  --use_distill --stu_iter=1 --vis_dir='test' --mask --triple_clues --text_focus --lca
# resume just only needs to be filled in the folder such as 'ckpt/rstn-1/'
