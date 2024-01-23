import os
import torch
import argparse
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms as tfms
from diffusers import StableDiffusionInpaintPipeline
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from util import *


# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
LOW_RESOURCE = False 
torch.manual_seed(1)

auth_token='your_auth_token'

# Load model
# #your files or CompVis/stable-diffusion-v1-4
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-4", use_auth_token = auth_token)
pipe=pipe.to(torch_device)
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
vae_magic = 0.18215
scheduler = LMSDiscreteScheduler(beta_start = 0.00085, beta_end = 0.012, 
                                 beta_schedule = "scaled_linear", num_train_timesteps = 1000)

#your files or runwayml/stable-diffusion-inpainting
inpaint = StableDiffusionInpaintPipeline.from_pretrained('stable-diffusion-inpainting', 
                                                         revision = "fp16", torch_dtype = torch.float16, 
                                                         safety_checker = None, use_auth_token = auth_token)
inpaint = inpaint.to(torch_device)

EMBEDDING_LEN = min(tokenizer.model_max_length, text_encoder.config.max_position_embeddings)
num_inference_steps = 50 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = "./dataset_txt/ImageNet_list.txt", help = "folder to load image")
    parser.add_argument("--dataset_name", type = str, default = "ImageNet", help = "name of dataset(Imagen or ImageNet)")
    parser.add_argument("--dataset_path", type = str, default = "ImageNet_dataset_path/", help = "the path of dataset(Imagen or ImageNet)")
    parser.add_argument("--output_path", type = str, default = "./result_sdedit/", help = "folder to save output")
    parser.add_argument("--allresult_path", type = str, default = "total", help = "the path save all the result")
    parser.add_argument("--strength", type = float, default = 0.5, help = "noise strength")
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
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs


@torch.no_grad()
def main(args):
    total_path = args.output_path + args.allresult_path + '/'
    if not os.path.exists(total_path):
        os.makedirs(total_path)
    data_pair=get_data(args.input_path,args.dataset_name)
    for data in data_pair:
        ids=data[0]
        mask_path = args.output_path  + ids + '-mask.jpg'
        if os.path.exists(mask_path): continue
        prompt = data[-1]
        if len(ids)==5:
            image_path = args.dataset_path+data[1]
        else:
            image_path = args.dataset_path+data[0]+'.jpg'
        input_image = Image.open(image_path).resize((512, 512)).convert('RGB')
        latent = image2latent(input_image)
        
        if args.or_save:
            input_image.save(args.output_path  + ids + '-or_image.jpg') # input image

        text_embeddings = get_text_embedding(tokenizer, text_encoder, torch_device, EMBEDDING_LEN, prompt)

        torch.manual_seed(torch.seed() if args.seed == None else args.seed)
        seed_list = torch.randint(0, 2**62, (1, )).tolist()
        
        scheduler.set_timesteps(num_inference_steps)
        offset = scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * args.strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * 1 * 1, device = torch_device)
        torch.manual_seed(seed_list[0])
        noises = torch.randn_like(latent) #+ 0.1 * torch.randn(latent.shape[0], latent.shape[1], 1, 1).cuda()
        latents = scheduler.add_noise(latent, noises, timesteps = timesteps)
        latents = latents.to(torch_device).float()
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = scheduler.timesteps[t_start:].to(torch_device)

        for tm in tqdm(timesteps):
            #predict noise
            latent_model_input = torch.cat([latents] * 2)#doing classifier free guidance
            latent_model_input = scheduler.scale_model_input(latent_model_input, tm)
            with torch.no_grad():
                noise_pred = unet(latent_model_input.type(unet.dtype), tm, encoder_hidden_states = text_embeddings.type(unet.dtype))["sample"]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, tm, latents).prev_sample

        img = latents2images (latents)[0]
        img.save(args.output_path  + ids + '-result.jpg')
        img.save(total_path + ids + '-result.jpg')
        


if __name__ == "__main__":    
    args = get_args()
    with torch.no_grad():
        main(args)
