import argparse
def get_args():
    parser = argparse.ArgumentParser(description='attack_alignment')
    parser.add_argument('--gpu', type=str, default='6', help='gpu id')
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--alpha', type=float, default=0.04, help='alpha')
    parser.add_argument('--niters', type=int, default=10, help='niters')
    parser.add_argument('--eps', type=float, default=0.4, help='eps')
    parser.add_argument('--sd_name', type=str, default='runwayml/stable-diffusion-v1-5', help='sd version,runwayml/stable-diffusion-v1-5,stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--data_path', type=str, default='data.json', help='image path')
    parser.add_argument('--lora_path', type=str, default='ckpt_2025', help='lora path')
    parser.add_argument('--output_path', type=str, default='result_20250518', help='output path')
    parser.add_argument('--source_model', type=str, default='ResNet50', help='')
    parser.add_argument('--use_lora', action='store_true', default=True,help='use lora or not')
    parser.add_argument('--targeted', action='store_true', default=False, help='targeted attack or not')
    parser.add_argument('--use_noise_optim', action='store_true',default=True, help='use noise optimization or not')
    parser.add_argument('--test_sample_num', type=int, default=1000, help='use guidance or not')
    parser.add_argument('--index_cond', default=40, type=int, help='use guidance or not')
    parser.add_argument('--use_da', action='store_true',default=True, help='use diffusion_augmentation')
    parser.add_argument('--gradient_backpop', default='skip-step', type=str, help='skip-step,grad-checkpoint')
    parser.add_argument('--inversion_step', default=10, type=int, help='use guidance or not')
    parser.add_argument('--save_image',default=True,type=bool)
    parser.add_argument('--attack_method', type=str, default='Ours_vca_p0.1', help='attack method')
    args = parser.parse_args()
    return args
args=get_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from pipe_new import AttackPipeline
from utils import *
from PIL import Image
import torchvision.transforms as transforms
import json
import csv
import torch.nn.functional as F
import time
import csv
from torchvision.utils import save_image
from datetime import datetime
import numpy as np
import random
now = datetime.now()
today_string = now.strftime("%m-%d|%H-%M")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)
    target_model_names=['mnv2','inception_v3', 'ResNet50', 'DenseNet161','ResNet152','EF-b7', 'mvit','vit','swint','pvtv2']
    source_model_names=[args.source_model]
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    img_size = 224
    if args.source_model == 'vit':
        mean = [0.5, 0.5, 0.5]
        stddev = [0.5, 0.5, 0.5]
    elif args.source_model == 'mvit':
        mean = [0, 0, 0]
        stddev = [1, 1, 1] 
        img_size = 320
    else:
        mean = [0.485, 0.456, 0.406]
        stddev = [0.229, 0.224, 0.225]
    mean_list=[]
    stddev_list=[]
    
    for name in target_model_names:
        if name == 'vit':
            mean_list.append([0.5, 0.5, 0.5])
            stddev_list.append([0.5, 0.5, 0.5])
        elif name == 'mvit':
            mean_list.append([0, 0, 0])
            stddev_list.append([1, 1, 1])
            img_size_mvit = 320
        else:
            mean_list.append([0.485, 0.456, 0.406])
            stddev_list.append([0.229, 0.224, 0.225])
    trn = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),])
    #load data
    with open(args.data_path) as f:
        data = json.load(f)
        
    total_img_num=args.test_sample_num
    data=data[:total_img_num]
    device='cuda'
    transfer_models = [WrapperModel(load_model(target_model_names[i]), mean_list[i], stddev_list[i]).to(device) for i in range(len(target_model_names))]
    detals='dual_path'+str(args.use_noise_optim)+'diffusion_augmentation_'+str(args.use_da)+'_gradient_backpop_'+args.gradient_backpop+'_use_lora_'+str(args.use_lora)
    attack_methods=args.attack_method
    reuslt_dir=os.path.join(args.output_path,attack_methods,args.source_model,detals)
    print(reuslt_dir)
    if not os.path.exists(reuslt_dir):
        os.makedirs(reuslt_dir)
    
    excel_path=os.path.join(reuslt_dir,'res'+'.csv')
    for model_i, source_model_name in enumerate(source_model_names):
        print(source_model_name)
        torch.cuda.empty_cache()
        batch_size=1
        # load models
        source_model = WrapperModel(load_model(source_model_name), mean, stddev).to(device)
        source_model = source_model.eval()
        print('Source model is loaded',flush=True)
        tot_time=0.
        niters=args.niters
        eps=args.eps
        alpha=args.alpha
        attack_config={'alpha':alpha,'niters':niters,'eps':eps}
        def iter_source():
            num_images = 0
            target_succs = {m: {attack_methods: ([0.] * (1))} for m in target_model_names}
            num_batches = int(np.ceil(len(data) / batch_size))
            total_time=0.
            for k in range(0,num_batches):
                batch_size_cur = 1
                img = torch.zeros(batch_size_cur,3,img_size,img_size).to(device)
                for i in range(batch_size_cur):
                    image_path=data[k * batch_size + i]['image_path']         
                    img[i] = trn(Image.open(image_path).convert('RGB'))
                source_model.eval()
                labels = torch.tensor([data[k]['label']]).to(device)
                if image_path.split('/')[-1] in os.listdir(reuslt_dir):
                    adv_image_tensor=torch.zeros(batch_size_cur,3,img_size,img_size).to(device)
                    adv_image_tensor[0]=trn(Image.open(os.path.join(reuslt_dir,image_path.split('/')[-1])).convert('RGB'))
                    num_images += batch_size_cur
                else:
                    prompt=data[k]['class']
                    pipeline: AttackPipeline = AttackPipeline.from_pretrained(args.sd_name).to("cuda")
                    num_images += batch_size_cur
                    if args.use_lora:
                        lora_path=os.path.join(args.lora_path,str(data[k]['id'])+'.safetensor')
                        pipeline.load_lora_weights(lora_path, adapter_name='a')
                    
                    
                    ori_latents=pipeline.enimg2latent(image_path)
                    
                    if args.gradient_backpop == 'skip-step':
                        st=time.time()
                        latent_T = pipeline.inverse(image_path, prompt, 50, guidance_scale=1)
                        adv_image_tensor = pipeline.attack_optimization(prompt=prompt, latents=latent_T,
                                guidance_scale=1.0,classfier=source_model,label=labels,attack_config=attack_config,use_noise_opt=args.use_noise_optim,index_cond=args.index_cond,
                                use_da=args.use_da,ori_latents=ori_latents,image_size=img_size)
                        et=time.time()
                        print('Time:',et-st)
                    elif args.gradient_backpop == 'vca':
                        st=time.time()
                        latent_T = pipeline.inverse(image_path, prompt, 50, guidance_scale=1)
                        adv_image_tensor = pipeline.vca(prompt=prompt, latents=latent_T,guidance_scale=1.0,image_size=img_size)
                        et=time.time()
                        print('Time:',et-st)
                    else:
                        start=time.time()
                        latent_T = pipeline.inverse(image_path, prompt, 50, guidance_scale=1,inversion_step=args.inversion_step)
                        adv_image_tensor = pipeline.attack_optimization_checkpoint(prompt=prompt, latents=latent_T,
                                guidance_scale=1.0,classfier=source_model,label=labels,attack_config=attack_config,use_noise_opt=args.use_noise_optim,index_cond=args.index_cond, image_size=img_size,
                                use_da=args.use_da,ori_latents=ori_latents,inversion_step=args.inversion_step)
                        end=time.time()
                        print('Time:',end-start)
                        
                    if args.save_image:
                        save_image(adv_image_tensor.clone(),os.path.join(reuslt_dir,image_path.split('/')[-1]))
                output_dict={attack_methods:adv_image_tensor.detach()}
                fin=0.
                for j, mod in enumerate(transfer_models):
                    mod.eval()
                    adv_img=adv_image_tensor.clone()
                    if target_model_names[j] == 'mvit':
                        adv_img=F.interpolate(adv_image_tensor.clone(), size=(img_size_mvit, img_size_mvit), mode='bilinear', align_corners=False)
                    elif target_model_names[j] == 'inception_v3':
                        adv_img=F.interpolate(adv_image_tensor.clone(), size=(299, 299), mode='bilinear', align_corners=False)
                    else:
                        if source_model_name == 'mvit':
                            adv_img=F.interpolate(adv_image_tensor.clone(), size=(224, 224), mode='bilinear', align_corners=False)
                        else:
                            pass
                    for n in range(1):
                        with torch.no_grad():
                            transfer_results_dict = {key: mod(adv_img).max(1)[1] for key, value in
                                                    output_dict.items()}
                        if not args.targeted:
                            target_succs[target_model_names[j]][attack_methods][n] += (
                                    torch.sum((transfer_results_dict[attack_methods] != labels).float())).item()
                        succ = (target_succs[target_model_names[j]][attack_methods][0]) / num_images
                        fin+=succ
                        print(f'[{k}/{len(data) }]Success Rate (%) on {target_model_names[j]} with {attack_methods} : {succ*100:.2f}',flush=True)
                print(f'[{k * batch_size+batch_size_cur}/{len(data) }]Success Rate (%) on average : {fin*10:.2f}',flush=True)
            return target_succs,total_time

 
        
        print(f"batch={batch_size}",flush=True)
        target_succs,tot_time = iter_source()
        print(datetime.now().strftime("%m-%d|%H-%M"),flush=True)
        with open(excel_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([' ',' ']+target_model_names+['avg']) 
            for a in [attack_methods]:  # Export experimental results
                for n in range(1):
                    res=[source_model_name,a]
                    aa=0.
                    for j, mod in enumerate(transfer_models):
                        final_succ = (target_succs[target_model_names[j]][a][n]) / total_img_num
                        res.append(f"{final_succ * 100:.2f}")
                        aa+=final_succ
                    res.append(f"{aa / len(target_model_names):.2f}")
                    
                    writer.writerow(res)
            


        print('AVG TIME: ',tot_time/total_img_num)
        print(datetime.now().strftime("%m-%d|%H-%M"),flush=True)
        
if __name__=='__main__':
    args=get_args()
    main(args)
