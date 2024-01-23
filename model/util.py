from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import copy


# read Imagen or ImageNet dataset
def get_data(path,dataset_name,index=2):
    if dataset_name not in ['ImageNet','Imagen','Editing-Mask']:
        print('error dataset name!')
        return None
    data_pair=[]
    with open(path,'r',encoding='utf-8') as f:
        for i in f.readlines():
            i=i.strip('\n')
            l=i.split(' <> ')
            if dataset_name=='ImageNet':
                data_pair.append([l[0][18:-5]]+l[:2]+['a '+l[-1]])
            elif dataset_name=='Imagen':
                data_pair.append([l[0]]+[l[1]+'_'+str(index)+'.jpg']+[l[-2]]+[l[-1]])
            elif dataset_name=='Editing-Mask':
                if len(l)==4:
                    data_pair.append([l[0]]+[l[1]+'_'+str(index)+'.jpg']+[l[-2]]+[l[-1]])
                else:
                    data_pair.append([l[0][18:-5]]+l[:2]+['a '+l[-1]])
        print("total numbers is : "+str(len(data_pair)))
    return data_pair

# def get_data(path,dataset_name,index=2):
#     if dataset_name not in ['ImageNet','Imagen']:
#         print('error dataset name!')
#         return None
#     data_pair=[]
#     with open(path,'r',encoding='utf-8') as f:
#         for i in f.readlines():
#             i=i.strip('\n')
#             l=i.split(' <> ')
#             if dataset_name=='ImageNet':
#                 data_pair.append(l[:2]+['a '+l[-1]])
#             elif dataset_name=='Imagen':
#                 data_pair.append([l[1]+'_'+str(index)+'.jpg']+[l[2]]+[l[-1]])
#         print("total numbers is : "+str(len(data_pair)))
#     return data_pair


#latent process
def init_latent(latent, scheduler, strength, seed_list, batch_size, torch_device, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * 1 * 1, device = torch_device)

    latents = []
    noises = []
    for ii in range(batch_size):
        torch.manual_seed(seed_list[ii])
        noise = torch.randn_like(latent) #+ 0.1 * torch.randn(latent.shape[0], latent.shape[1], 1, 1).cuda()
        latent0 = scheduler.add_noise(latent, noise, timesteps = timesteps)
        latents.append(latent0)
        noises.append(noise)
    latents = torch.cat(latents)
    noises = torch.cat(noises)
    latents = latents.to(torch_device).float()
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = scheduler.timesteps[t_start:].to(torch_device)
    return latents, noises, timesteps

def get_or_latent(latent, noises, tm, batch_size, scheduler, torch_device):
    or_latents = []
    timestep = torch.tensor([tm] * 1 * 1, device = torch_device)
    for j in range(batch_size):
        scheduler_new = copy.deepcopy(scheduler)
        noise = noises[j]  
        or_latent = scheduler_new.add_noise(latent, noise, timesteps = timestep)
        or_latents.append(or_latent)
    or_latents = torch.cat(or_latents)
    return or_latents



# show the image
def concat_images_all(COL,ROW,image_files,path,name,w=64,h=64,SAVE_QUALITY=100):
    nums=len(image_files)
    target = Image.new('RGB', (w * COL, h * ROW))
    for row in range(ROW):
        for col in range(COL):
            if COL*row+col<nums:
                target.paste(image_files[COL*row+col], (0 + w*col, 0 + h*row))
    target.save(path + name + '.jpg', quality=SAVE_QUALITY) 



# calculate the consine similarity between attention map
def cal_sim(images, base_index = 0):
    base = np.array(images[base_index]).flatten()
    image = images
    image.pop(base_index)
    scros = []
    max_scros = 0
    for i, im in enumerate(image):
        im = np.array(im).flatten()
        sims = cosine_similarity(im.reshape(1, -1), base.reshape(1, -1))
        scros.append(float(sims[0]))
        if sims > max_scros:
            max_scros = sims
            max_index = i + 1
    scros.insert(base_index, 1.0)
    return max_index, scros


#get the text embedding
def get_embedding_for_prompt(prompt, tokenizer, text_encoder, torch_device, max_len):
    max_length = max_len
    tokens = tokenizer([prompt], padding = "max_length", max_length = max_length, truncation = True, return_tensors = "pt")
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(torch_device))[0]
    return embeddings

def get_text_embedding(tokenizer, text_encoder, torch_device, max_len, text1, text2 = '', batch_size = 1):
    emb = get_embedding_for_prompt(text1, tokenizer, text_encoder, torch_device, max_len).cpu()
    uncond = get_embedding_for_prompt(text2, tokenizer, text_encoder, torch_device, max_len).cpu()
    emb = np.array(emb).astype(np.float16)
    emb = torch.from_numpy(emb)
    embs = emb.expand(batch_size, -1, -1).to('cuda')
    uncond = np.array(uncond).astype(np.float16)
    uncond = torch.from_numpy(uncond)
    unconds = uncond.expand(batch_size, -1, -1).to('cuda')
    text_embeddings = torch.cat([unconds, embs])
    return text_embeddings    












