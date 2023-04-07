import copy
import glob
import os
import time
import pprint
from collections import deque
from collections import OrderedDict
from torchvision.utils import save_image
from torchvision import transforms

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model1 import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.function import AverageMeter
from evaluation import evaluate

from tensorboardX import SummaryWriter
from utils.utils import create_logger
from utils.utils import save_checkpoint
from a2c_ppo_acktr.config import config
from a2c_ppo_acktr.config import update_config
from a2c_ppo_acktr.config import update_dir
from a2c_ppo_acktr.config import get_model_name
from PIL import Image
import cv2
from baselines.common.atari_wrappers import WarpFrame

def main():
    args = get_args()     #添加命令行，使用配置文件yaml更换config参数

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True   #保持卷积算法的确定性，便于复现

    logger, final_output_dir, tb_log_dir = create_logger(   #日志，输出文件名，tensorboard文件名
        config, args.cfg, 'train', seed=config.seed) 

    eval_log_dir = final_output_dir + "_eval"

    utils.cleanup_log_dir(final_output_dir)
    utils.cleanup_log_dir(eval_log_dir)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer = SummaryWriter(tb_log_dir)  #将summary写入tb_log_dir中

    torch.set_num_threads(1)  #线程数，加快进程时调整
    device = torch.device("cuda:" + config.GPUS if config.cuda else "cpu")

    width = height = 84
    envs = make_vec_envs(config.env_name, config.seed, config.num_processes,    
                         config.gamma, final_output_dir, device, False,
                         width=width, height=height, ram_wrapper=False)   #PongNoFrameskip-v4, 1 ,16, 0.99, dir, cuda0, 84, 84, ram=false
    # create agent
    #print(envs.observation_space.shape, envs.action_space)
    actor_critic = Policy(
        device,
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': config.recurrent_policy,
                     'hidden_size': config.hidden_size,
                     'feat_from_selfsup_attention': config.feat_from_selfsup_attention,
                     'feat_add_selfsup_attention': config.feat_add_selfsup_attention,
                     'feat_mul_selfsup_attention_mask': config.feat_mul_selfsup_attention_mask,
                     'selfsup_attention_num_keypoints': config.SELFSUP_ATTENTION.NUM_KEYPOINTS,
                     'selfsup_attention_gauss_std':  config.SELFSUP_ATTENTION.GAUSS_STD,
                     'selfsup_attention_fix': config.selfsup_attention_fix,
                     'selfsup_attention_fix_keypointer': config.selfsup_attention_fix_keypointer,
                     'selfsup_attention_pretrain': config.selfsup_attention_pretrain,
                     'selfsup_attention_keyp_maps_pool': config.selfsup_attention_keyp_maps_pool,
                     'selfsup_attention_image_feat_only': config.selfsup_attention_image_feat_only,
                     'selfsup_attention_feat_masked': config.selfsup_attention_feat_masked,
                     'selfsup_attention_feat_masked_residual': config.selfsup_attention_feat_masked_residual,
                     'selfsup_attention_feat_load_pretrained': config.selfsup_attention_feat_load_pretrained,
                     'use_layer_norm': config.use_layer_norm,
                     'selfsup_attention_keyp_cls_agnostic': config.SELFSUP_ATTENTION.KEYPOINTER_CLS_AGNOSTIC,
                     'selfsup_attention_feat_use_ln': config.SELFSUP_ATTENTION.USE_LAYER_NORM,
                     'selfsup_attention_use_instance_norm': config.SELFSUP_ATTENTION.USE_INSTANCE_NORM,
                     'feat_mul_selfsup_attention_mask_residual': config.feat_mul_selfsup_attention_mask_residual,
                     'bottom_up_form_objects': config.bottom_up_form_objects,
                     'bottom_up_form_num_of_objects': config.bottom_up_form_num_of_objects,
                     'gaussian_std': config.gaussian_std,
                     'train_selfsup_attention': config.train_selfsup_attention,
                     'block_selfsup_attention_grad': config.block_selfsup_attention_grad,
                     'sep_bg_fg_feat': config.sep_bg_fg_feat,
                     'mask_threshold': config.mask_threshold,
                     'fix_feature': config.fix_feature,
                     'num_processes': config.num_processes,
                     'temperature':config.temperature,
                     'train_selfsup_attention_batch_size': config.train_selfsup_attention_batch_size,
                     })

    # init / load parameter
    if config.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.MODEL_FILE))
        state_dict = torch.load(config.MODEL_FILE)

        state_dict = OrderedDict((_k, _v) for _k, _v in state_dict.items() if 'dist' not in _k)

        actor_critic.load_state_dict(state_dict, strict=False)
    elif config.RESUME:
        checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint.pth'
        )
        if os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            actor_critic.load_state_dict(checkpoint['state_dict'])

            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))

    actor_critic.to(device)

    if config.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            alpha=config.alpha,
            max_grad_norm=config.max_grad_norm,
            train_selfsup_attention=config.train_selfsup_attention)
    elif config.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            config.clip_param,
            config.ppo_epoch,
            config.num_mini_batch,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm)
    elif config.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, config.value_loss_coef, config.entropy_coef, acktr=True,
            train_selfsup_attention=config.train_selfsup_attention,
            max_grad_norm=config.max_grad_norm
        )

    # rollouts: environment
    rollouts = RolloutStorage(config.num_steps, config.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              keep_buffer=config.train_selfsup_attention,
                              buffer_size=config.train_selfsup_attention_buffer_size) #5, 16， 

    if config.RESUME:
        if os.path.exists(checkpoint_file):
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
    obs = envs.reset()
    
    
    #rollouts.obs[0].copy_(obs)    #初始值赋值
    rollouts.to(device)
    env_test = gym.make('CarnivalNoFrameskip-v4')
    env2 = WarpFrame(env_test, width=84, height=84)
    frame = env_test.reset()
    frame2 = env2.reset()
    print(obs.shape)  
    save_dir = 'graph2'
    for i in range(167):
        
   
        pil_image = Image.fromarray(frame)
        
        pil_image.save("graph/" + str(i) + "_frame.png")  
        save_path = os.path.join(save_dir, str(i) + "_frame.png")
        cv2.imwrite(save_path, frame2)
      
        #transforms.ToPILImage()(obs[:, 0, :, :]).save("graph/" + str(i) + "_frame.png")
        #cv2.imwrite('gray'+str(i)+'_image.png', obs[:, 0, :, :])
        #save_image(obs[:, 0, :, :], "graph/" + str(i) + "_frame.png")
        
        tensor_img = torch.from_numpy(frame2).permute(2, 0, 1).float().cuda()
        
        
        value, action, action_log_prob = actor_critic.act(
                        tensor_img.unsqueeze(0))
        
        a = env_test.action_space.sample()
        frame, _, _, _ = env_test.step(a)
                # Obser reward and next obs
        frame2, _, _, _ = env2.step(a)
        
        
        #rollouts.insert(frame)

    episode_rewards = deque(maxlen=10)

    print('done')    
    
    '''resized_img = cv2.resize(frame, (84,84))

print(resized_img.shape)
# 将NumPy数组转换为Torch张量
tensor_img = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0


# 创建一个(84,84,3)的NumPy数组
image = np.zeros((84,84,3), dtype=np.uint8)

# 创建PIL图像对象并保存到文件
pil_image = Image.fromarray(image)
pil_image.save('example_image.jpg')'''

'''for i in range(100):
    frame, _, _, _ = env.step(env.action_space.sample())
    for i in range(167):
        if i > 164:
            print('ii', obs[:, 0, :, :])
        transforms.ToPILImage()(obs[:, 0, :, :]).save("graph/" + str(i) + "_frame.png")
        #cv2.imwrite('gray'+str(i)+'_image.png', obs[:, 0, :, :])
        #save_image(obs[:, 0, :, :], "graph/" + str(i) + "_frame.png")
        i = i % 5
        
        value, action, action_log_prob = actor_critic.act(
                        obs)

                # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)
        
        rollouts.insert(obs)'''

    


if __name__ == "__main__":
    main()
