from droid.evaluation.eval_launcher_robomimic import eval_launcher, get_goal_im
import matplotlib.pyplot as plt
import os
import argparse
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--capture_goal', action='store_true')
    parser.add_argument('-l', '--lang_cond', action='store_true')
    parser.add_argument('-c', '--ckpt_path', type=str, default=None, 
        help='Path to Pytorch checkpoint (.pth) corresponding to the policy you want to evaluate.')
    args = parser.parse_args()

    variant = dict(
        exp_name="policy_test",
        save_data=False,
        use_gpu=True,
        seed=0,
        policy_logdir="test",
        task="",
        layout_id=None,
        model_id=50,
        camera_kwargs=dict(),
        data_processing_kwargs=dict(
            timestep_filtering_kwargs=dict(),
            image_transform_kwargs=dict(),
        ),
        ckpt_path=args.ckpt_path,
    )
    
    print("Evaluating Policy")
    eval_launcher(variant, run_id=1, exp_id=0)
