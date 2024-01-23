import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms as tfms
from transformers import CLIPTextModel,CLIPTokenizer
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from diffusers import AutoencoderKL,UNet2DConditionModel,LMSDiscreteScheduler,StableDiffusionInpaintPipeline

from util import *
# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

auth_token='your_auth_token'
torch.manual_seed(1)
num_inference_steps = 50 

# Load model
#your files or CompVis/stable-diffusion-v1-4
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token = auth_token)
pipe=pipe.to(torch_device)
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
vae_magic = 0.18215
scheduler = LMSDiscreteScheduler(beta_start = 0.00085, beta_end = 0.012, 
                                 beta_schedule = "scaled_linear", num_train_timesteps = 1000)

#your files or runwayml/stable-diffusion-inpainting
inpaint = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting', 
                                                         revision = "fp16", torch_dtype = torch.float16, 
                                                         safety_checker = None, use_auth_token = auth_token)
inpaint = inpaint.to(torch_device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = "./dataset_txt/ImageNet_list.txt", help = "folder to load image")
    parser.add_argument("--dataset_name", type = str, default = "ImageNet", help = "name of dataset(Imagen or ImageNet)")
    parser.add_argument("--dataset_path", type = str, default = "ImageNet_dataset_path/", help = "the path of dataset(Imagen or ImageNet)")
    parser.add_argument("--output_path", type = str, default = "./result_diffedit/", help = "folder to save output")
    parser.add_argument("--allresult_path", type = str, default = "total", help = "the path save all the result")
    parser.add_argument("--strength", type = float, default = 0.5, help = "noise strength")
    parser.add_argument("--threshold", type = float, default = 0.3, help = "threshold for binarization")
    parser.add_argument("--or_save", type = str, default = True, help = "if save the original image in output path")
    parser.add_argument("--seed", type = int, default = 4623523532252, help = "seed values")
    args, _ = parser.parse_known_args()
    print(args)
    return args 

def image2latent(im):
    im = tfms.ToTensor()(im).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(im.to(torch_device)*2-1)
    latent = latent.latent_dist.sample() * vae_magic      
    return latent
    
def latents2images(latents):
    latents = latents/vae_magic
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0,1)
    imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs
    
def get_embedding_for_prompt(prompt):
    max_length = tokenizer.model_max_length
    tokens = tokenizer([prompt],padding="max_length",max_length=max_length,truncation=True,return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(torch_device))[0]
    return embeddings

# Given a starting image latent and a prompt; predict the noise that should be removed to transform
# the noised source image to a denoised image guided by the prompt.
def predict_noise(text_embeddings,im_latents,strength=0.5,seed=torch.seed(),guidance_scale=7.5,**kwargs):
    num_inference_steps = 50            # Number of denoising steps
    torch.manual_seed(seed)   # Seed generator to create the inital latent noise
    uncond = get_embedding_for_prompt('')
    text_embeddings = torch.cat([uncond, text_embeddings])
    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)
    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * 1 * 1, device=torch_device)
    noise = torch.randn_like(im_latents)
    latents = scheduler.add_noise(im_latents,noise,timesteps=timesteps)
    latents = latents.to(torch_device).float()
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = scheduler.timesteps[t_start:].to(torch_device)
    noise_pred = None
    for i, tm in enumerate(timesteps):#25次
        latent_model_input = torch.cat([latents] * 2)#doing classifier free guidance
        latent_model_input = scheduler.scale_model_input(latent_model_input, tm)

        # predict the noise residual
        with torch.no_grad():#float 32     2/4/64/64                            2/77/768
            noise_pred = unet(latent_model_input, tm, encoder_hidden_states=text_embeddings)["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        u = noise_pred_uncond
        g = guidance_scale
        t = noise_pred_text
        # perform guidance
        noise_pred = u + g * (t - u)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, tm, latents).prev_sample

    return latents2images(latents)[0],noise_pred#取得到的图片和最后x1->x0这步预测的噪声


# For a given reference prompt and a query prompt;
# Run the diffusion process 10 times; Calculating a "noise distance" for each sample
def calc_diffedit_samples(encoded,prompt1,prompt2,strength,n=10,**kwargs):
    diffs=[]
    # So we can reproduce mask generation we generate a list of n seeds
    torch.manual_seed(torch.seed() if 'seed' not in kwargs else kwargs['seed'])
    seeds = torch.randint(0,2**62,(n,)).tolist()
    for i in range(n):
        kwargs['seed'] = seeds[i] # Important to use same seed for the two noise samples
        emb1 = get_embedding_for_prompt(prompt1)
        _im1,n1 = predict_noise(emb1,encoded,strength,**kwargs)
        emb2 = get_embedding_for_prompt(prompt2)
        _im2,n2 = predict_noise(emb2,encoded,strength,**kwargs)
        diffs.append((n1-n2)[0].pow(2).sum(dim=0).pow(0.5)[None])
    all_masks = torch.cat(diffs)
    return all_masks

# Given an image latent and two prompts; generate a grayscale diff by sampling the noise predictions
# between the prompts.
def calc_diffedit_diff(im_latent,p1,p2,strength,**kwargs):
    m = calc_diffedit_samples(im_latent,p1,p2,strength,**kwargs)
    m = m.mean(axis=0) # average samples together
    m = (m-m.min())/(m.max()-m.min()) # rescale to interval [0,1]
    m = (m*255.).cpu().numpy().astype(np.uint8)
    m = Image.fromarray(m)
    return m

# Try to improve the mask thru convolutions etc
# assume m is a PIL object containing a grayscale 'diff'
def process_diffedit_mask(m,threshold=0.35,**kwargs):
    m = np.array(m).astype(np.float32)
    m = cv2.GaussianBlur(m,(5,5),1)
    m = (m>(255.*threshold)).astype(np.float32)*255
    m = Image.fromarray(m.astype(np.uint8))
    return m






import time 
def main(args):
    total_path = args.output_path + args.allresult_path + '/'
    if not os.path.exists(total_path):
        os.makedirs(total_path)
    data_pair=get_data(args.input_path,args.dataset_name)
    for data in data_pair:
        ids=data[0]
        prompts = [data[-2],data[-1]]

        if len(ids)==5:
            image_path = args.dataset_path+data[1]
        else:
            image_path = args.dataset_path+data[0]+'.jpg'
        input_image = Image.open(image_path).resize((512,512)).convert('RGB')
        latent = image2latent(input_image)

        if args.or_save:
            input_image.save(args.output_path + ids + '-or_image.jpg')

        torch.manual_seed(torch.seed() if args.seed == None else args.seed)
        seed_list = torch.randint(0, 2**62, (1, )).tolist()
 
        mask = calc_diffedit_diff(latent,prompts[0],prompts[-1],args.strength,seed=seed_list[0])
        mask.save(args.output_path + ids + '-mask.png')
        binarized_mask = process_diffedit_mask(mask,threshold=args.threshold)
        binarized_mask.save(args.output_path + ids + '-mask_bi.png')
        binarized_mask = binarized_mask.resize((512,512))

        generator = torch.Generator(torch_device).manual_seed(args.seed)
        im_result = inpaint(prompt=[prompts[-1]],image=input_image,mask_image=binarized_mask,generator=generator).images[0]
        im_result.save(args.output_path + ids + '-inpainting.jpg')
        im_result.save(total_path + ids + '-inpainting.jpg')


if __name__ == "__main__":    
    args = get_args()
    with torch.no_grad():
        main(args)
