import os
import cv2
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from PIL import Image, ImageChops
from torchvision import transforms as tfms
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline

from util import *
from attention_store import *

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
LOW_RESOURCE = False 
torch.manual_seed(1)

auth_token='your_auth_token'


# Load model
#your files or CompVis/stable-diffusion-v1-4
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
    parser.add_argument("--input_path", type = str, default = "./image/rubby.jpg", help = "folder to load image")
    parser.add_argument("--output_path", type = str, default = "./", help = "folder to save output")
    parser.add_argument("--prompt", type = str, default = "a photo of a cat", help = "edit text")
    parser.add_argument("--output_name", type = str, default = "inpainting", help = "output name")
    parser.add_argument("--strength", type = float, default = 0.5, help = "noise strength")
    parser.add_argument("--threshold", type = float, default = 0.3, help = "threshold for binarization")
    parser.add_argument("--batch_size", type = int, default = 3, help = "the average times for stablity")
    parser.add_argument("--mask_path", type = str, default = None, help = "mask path(if have, will not generate mask later)")
    parser.add_argument("--or_save", type = str, default = True, help = "if save the original image in output path")
    parser.add_argument("--mask_save", type = str, default = True, help = "if save the generation mask in output path")
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

def use_mask(x_t, mask, or_latent):
    mask = tfms.ToTensor()(mask.resize((x_t.shape[-2], x_t.shape[-1])))
    mask = mask.expand(len(x_t[0]), -1, -1).unsqueeze(0)
    x_t = x_t.cpu()
    mask = mask.float()
    or_latent = or_latent.cpu()
    x_t = or_latent + mask * (x_t - or_latent)
    return x_t.cuda()

def get_position(images, position = [],th1=0.9,th2=0.3):
    indexx, _ = cal_sim(images[:-1])
    _, sorces  = cal_sim(images[1:-1], indexx-1)
    position.append(0)
    for inst in sorces:
        if inst >= th1:
            position.append(1)
        elif inst <= th2:
            position.append(-1)
        else:
            position.append(0)
    position.append(0)

    return position

def token_combination(tokens, prompt_list, attention_maps, lens = 64):
    flag = 0
    j = 0
    attention_token = ''
    attens = []
    images = []
    if len(tokens) > len(prompt_list) + 2:
        flag = 1
    if flag == 0:
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = Image.fromarray(image.numpy().astype(np.uint8)).resize((lens, lens))
            images.append(image)
    else:
        for i in range(len(tokens)):
            if i == 0 or i == len(tokens)-1:
                image = attention_maps[:, :, i]
            else:
                inst = tokenizer.decode(int(tokens[i]))
                if  inst == prompt_list[j]:
                    image = attention_maps[:, :, i]
                else:
                    attention_token = attention_token + inst
                    attens.append(attention_maps[:, :, i])
                    if attention_token.lower() == prompt_list[j].lower():
                        image = torch.stack(attens, 0).mean(0)
                        attention_token = ''
                        attens = []
                    else:
                        continue
                j = j + 1
            image = 255 * image / image.max()
            image = Image.fromarray(image.numpy().astype(np.uint8)).resize((lens, lens))
            images.append(image)
        
    for i in range(len(images)):
        if i == 0:
            images[i] = ImageChops.invert(images[i])
        bi_diff = tfms.ToTensor()(images[i])[0]
        bi_diff = (bi_diff-bi_diff.min())/(bi_diff.max()-bi_diff.min())*255.
        bi_diff = np.array(bi_diff).astype(np.uint8)
        bi_diff = Image.fromarray((bi_diff).astype(np.uint8))
        images[i] = bi_diff
    return images

def refine(images, position):
    back = None
    objects =  None
    back_number = sum(k == -1 for k in position)
    objects_number = sum(k == 1 for k in position)
    for inst in range(len(images)):
        if position[inst] == 1:
            if objects == None:
                objects = (tfms.ToTensor()(images[inst]))
            else:
                objects = objects + (tfms.ToTensor()(images[inst]))

        elif position[inst] == -1:
            if back == None:
                back = (tfms.ToTensor()(images[inst]))
            else:
                back = back + (tfms.ToTensor()(images[inst]))
    
    if back == None:
        objects = objects/objects_number
        objects = (objects-objects.min())/(objects.max()-objects.min())
        img = objects[0]
    else:
        back = back/back_number
        back = (back-back.min())/(back.max()-back.min())
        objects = objects/objects_number
        objects = (objects-objects.min())/(objects.max()-objects.min())
        img = (objects[0]-back[0])
        img = torch.clamp(img, min = 0.0)
    return img

def post_deal(img, th):
    img = np.array(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = (img-img.min())/(img.max()-img.min())
    mask_diff = Image.fromarray((img*255.).astype(np.uint8))
    img = (img-img.min())/(img.max()-img.min())
    img = np.array((img) > th).astype(np.float32)
    mask_image = Image.fromarray((img*255.).astype(np.uint8))
    return mask_diff, mask_image

def get_mask(position, batch_size, attention_store, prompts, tokenizer, th, res, from_where = ("up", "down")):
    tokens = tokenizer.encode(prompts[0])
    prompt_list = prompts[0].split(' ') 
    attention_maps = aggregate_attention(batch_size, attention_store, res, from_where, True, 0)
    images = token_combination(tokens, prompt_list, attention_maps)
    if len(images) == 3:
        img = (tfms.ToTensor()(images[1]))[0]
    else:
        if position == []:
            position = get_position(images, position)
        img = refine(images, position)
    mask_diff, mask_image = post_deal(img, th)
    
    return mask_diff, mask_image, position



@torch.no_grad()
def main(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    input_image = Image.open(args.input_path).resize((512, 512)).convert('RGB')
    latent = image2latent(input_image)

    if args.or_save:
        input_image.save(args.output_path + 'or_image.jpg') # input image

    if args.mask_path!= 'None':
        mask_image = Image.open(args.mask_path)
        generator = torch.Generator(torch_device).manual_seed(args.seed)
        im_result = inpaint(prompt = args.prompt, image = input_image, mask_image = mask_image.resize((512, 512)), generator = generator).images[0]
        im_result.save(args.output_path + args.output_name + '.jpg')
        return

    # ready for attntion store
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    # get text embedding and latents
    text_embeddings = get_text_embedding(tokenizer, text_encoder, torch_device, EMBEDDING_LEN, args.prompt, 
                                         batch_size = args.batch_size)

    torch.manual_seed(torch.seed() if args.seed == None else args.seed)
    seed_list = torch.randint(0, 2**62, (args.batch_size, )).tolist()
    latents, noises, timesteps = init_latent(latent, scheduler, args.strength, seed_list
                                             , args.batch_size, torch_device, num_inference_steps)
    position = []
    for tm in tqdm(timesteps):
        #get or_latents
        or_latents = get_or_latent(latent, noises, tm, args.batch_size, scheduler, torch_device)

        #predict noise
        latent_model_input = torch.cat([latents] * 2)#doing classifier free guidance
        latent_model_input = scheduler.scale_model_input(latent_model_input, tm)
        with torch.no_grad():
            noise_pred = unet(latent_model_input.type(unet.dtype), tm, encoder_hidden_states = text_embeddings.type(unet.dtype))["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, tm, latents).prev_sample

        # get mask from attention map
        _, mask_image, position = get_mask(position, args.batch_size, controller
                                               , [args.prompt], tokenizer, th = args.threshold, res = 16)

        # use mask
        latents = use_mask(latents, mask_image, or_latents)
    
    # inpainting 
    generator = torch.Generator(torch_device).manual_seed(args.seed)
    im_result = inpaint(prompt = args.prompt, image = input_image, mask_image = mask_image.resize((512, 512)), generator = generator).images[0]

    im_result.save(args.output_path + args.output_name + '.jpg')
    if args.mask_save:
        mask_image.save(args.output_path + 'mask.jpg')


        
if __name__ == "__main__":    
    args = get_args()
    main(args)
