# import requests
# import wget


# from argparse import Namespace
# import os
# import sys
# import numpy as np

# from PIL import Image

# import torch
# import torchvision.transforms as transforms



# pretrained_model_dir = os.path.join("/content", "models")
# os.makedirs(pretrained_model_dir, exist_ok=True)


# restyle_dir = os.path.join("/content", "restyle")
# stylegan_ada_dir = os.path.join("/content", "stylegan_ada")
# stylegan_nada_dir = os.path.join("/content", "stylegan_nada")

# output_dir = os.path.join("/content", "output")

# output_model_dir = os.path.join(output_dir, "models")
# output_image_dir = os.path.join(output_dir, "images")


# sys.path.append(restyle_dir)
# sys.path.append(stylegan_nada_dir)
# sys.path.append(os.path.join(stylegan_nada_dir, "ZSSGAN"))

# print(torch.__version__)
# print(restyle_dir)
# print(stylegan_ada_dir)
# print(stylegan_nada_dir)
# print(output_dir)
# print(output_model_dir)
# print(output_image_dir)

# device = 'cuda'

# dataset_sizes = {
#     "ffhq":   1024
# }

# model_names = {
#     "ffhq":   "ffhq.pt"
# }
                
# source_model_type = 'ffhq'               

# pt_file_name = 'ffhq.pt'

# # print(os.environ.get('CUDA_PATH'))


# from stylegan_nada.ZSSGAN.model.ZSSGAN import ZSSGAN

# from tqdm import notebook

# from stylegan_nada.ZSSGAN.utils.file_utils import save_images, get_dir_img_list
# from stylegan_nada.ZSSGAN.utils.training_utils import mixing_noise

# from IPython.display import display

# source_class = "Photo" #@param {"type": "string"}
# target_class = "Sketch" #@param {"type": "string"}

# style_image_dir = "dataset" #@param {'type': 'string'}

# target_img_list = get_dir_img_list(style_image_dir) if style_image_dir else None

# improve_shape = False #@param{type:"boolean"}

# print(source_class,target_class, style_image_dir, target_img_list)
# model_choice = ["ViT-B/32", "ViT-B/16"]
# model_weights = [1.0, 0.0]

# if improve_shape or style_image_dir:
#     model_weights[1] = 1.0
    
# mixing = 0.9 if improve_shape else 0.0

# auto_layers_k = int(2 * (2 * np.log2(dataset_sizes[source_model_type]) - 2) / 3) if improve_shape else 0
# auto_layer_iters = 1 if improve_shape else 0

# training_iterations = 151 #@param {type: "integer"}
# output_interval     = 50 #@param {type: "integer"}
# save_interval       = 0 #@param {type: "integer"}

# training_args = {
#     "size": dataset_sizes[source_model_type],
#     "batch": 2,
#     "n_sample": 4,
#     "output_dir": output_dir,
#     "lr": 0.002,
#     "frozen_gen_ckpt": os.path.join(pretrained_model_dir, pt_file_name),
#     "train_gen_ckpt": os.path.join(pretrained_model_dir, pt_file_name),
#     "iter": training_iterations,
#     "source_class": source_class,
#     "target_class": target_class,
#     "lambda_direction": 1.0,
#     "lambda_patch": 0.0,
#     "lambda_global": 0.0,
#     "lambda_texture": 0.0,
#     "lambda_manifold": 0.0,
#     "auto_layer_k": auto_layers_k,
#     "auto_layer_iters": auto_layer_iters,
#     "auto_layer_batch": 8,
#     "output_interval": 50,
#     "clip_models": model_choice,
#     "clip_model_weights": model_weights,
#     "mixing": mixing,
#     "phase": None,
#     "sample_truncation": 0.7,
#     "save_interval": save_interval,
#     "target_img_list": target_img_list,
#     "img2img_batch": 16,
#     "channel_multiplier": 2,
# }

# args = Namespace(**training_args)

# print("Loading base models...")
# net = ZSSGAN(args)
# print("Models loaded! Starting training...")

# g_reg_ratio = 4 / 5

# g_optim = torch.optim.Adam(
#     net.generator_trainable.parameters(),
#     lr=args.lr * g_reg_ratio,
#     betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
# )

# # Set up output directories.
# sample_dir = os.path.join(args.output_dir, "sample")
# ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

# os.makedirs(sample_dir, exist_ok=True)
# os.makedirs(ckpt_dir, exist_ok=True)

# seed = 3 #@param {"type": "integer"}

# torch.manual_seed(seed)
# np.random.seed(seed)

# # Training loop
# fixed_z = torch.randn(args.n_sample, 512, device=device)

# for i in notebook.tqdm(range(args.iter)):
#     net.train()
        
#     sample_z = mixing_noise(args.batch, 512, args.mixing, device)

#     [sampled_src, sampled_dst], clip_loss = net(sample_z)

#     net.zero_grad()
#     clip_loss.backward()

#     g_optim.step()

#     if i % output_interval == 0:
#         net.eval()

#         with torch.no_grad():
#             [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation)

#             if source_model_type == 'car':
#                 sampled_dst = sampled_dst[:, :, 64:448, :]

#             grid_rows = 4

#             save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

#             img = Image.open(os.path.join(sample_dir, f"dst_{str(i).zfill(6)}.jpg")).resize((1024, 256))
#             display(img)
    
#     if (args.save_interval > 0) and (i > 0) and (i % args.save_interval == 0):
#         torch.save(
#             {
#                 "g_ema": net.generator_trainable.generator.state_dict(),
#                 "g_optim": g_optim.state_dict(),
#             },
#             f"{ckpt_dir}/{str(i).zfill(6)}.pt",
#         )
