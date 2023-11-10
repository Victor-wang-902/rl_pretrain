import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

def normalize_np(embs):
    return F.normalize(torch.tensor(embs), p=2, dim=1).numpy()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_pooling_norm(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return F.normalize(embeddings, p=2, dim=1)

def weighted_pooling(emb, attention_mask, window_size=8, dropoff=0.):
    if emb.dim() == 2:
        emb_exp = emb.unsqueeze(0)
    else:
        emb_exp = emb.clone()
    mask = torch.full(emb_exp.shape, dropoff).to(attention_mask.device)
    mask[:,:window_size,:] = 1.
    mask = torch.cumprod(mask, dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(emb_exp.size()).float()
    mask = mask * input_mask_expanded
    #print(mask)
    emb_exp = torch.sum(emb_exp * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    #print(emb_exp)
    return F.normalize(emb_exp, p=2, dim=1)

#Encode text
def encode(inputs, args=None, model=None):
    # Compute token embeddings
    if args is not None:
        model = args.model
    with torch.no_grad():
        model_output = model(**inputs, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, inputs['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def encode_no_pool(inputs, args=None, model=None):
    # Compute token embeddings
    if args is not None:
        model = args.model
    with torch.no_grad():
        model_output = model(**inputs, return_dict=True)
    token_embeddings = model_output.last_hidden_state
    #input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    #embeddings = (token_embeddings * input_mask_expanded)

    # Normalize embeddings
    #embeddings = F.normalize(embeddings, p=2, dim=1)
    return token_embeddings


def init_model(args: argparse.Namespace):
    model = AutoModel.from_pretrained(args.model_name, is_decoder=False)
    return model

    
def pad_emb(emb_list, pad):
    max_len = max([emb.shape[0] for emb in emb_list])
    new_emb_list = [torch.concat([emb] + [pad] * (max_len-emb.shape[0]), dim=0) for emb in emb_list]
    return torch.stack(new_emb_list, dim=0)
    
##############################################################################################################
def preprocess_with_text(dataset, encoder, tokenizer, description):
    state_inputs = tokenizer(description["state"], padding=True, truncation=False, return_tensor="pt")
    action_inputs = tokenizer(description["action"], padding=True, truncation=False, return_tensor="pt")
    # Compute token embeddings
    with torch.no_grad():
        state_outputs = model(**state_inputs)
        action_outputs = model(**action_outputs)
    
    # Perform pooling
    state_outputs = mean_pooling(state_outputs, inputs['attention_mask'])
    action_outputs = mean_pooling(action_outputs, inputs['attention_mask'])

    # Normalize embeddings
    state_outputs = normalize_np(state_outputs)
    action_outputs = normalize_np(action_outputs)

    dataset["observation"] = np.concatenate([dataset["observation"], state_outputs], dim=1)
    dataset["next_observation"] = np.concatenate([dataset["next_observation"], state_outputs], dim=1)
    dataset["action"] = np.concatenate([dataset["action"], action_outputs], dim=1)
    return dataset
###############################################################################################################





    

