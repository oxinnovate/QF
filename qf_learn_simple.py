#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,shutil
import json
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

model_path = "/home/ls/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct"
qflayer=23
qffolder="/home/ls/Github/ChatGLM3-main/QF/parameters"


# 过滤警告信息
warnings.filterwarnings("ignore")
# 设置随机种子确保结果可重复
qfseed=1
torch.manual_seed(qfseed)
torch.cuda.manual_seed(qfseed)
torch.cuda.manual_seed_all(qfseed)
np.random.seed(qfseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch

def compute_W_prime(W, u, up, v, vp):

    X=u
    Y=up
    B=v-vp

    """
    计算 W' = W - [W(Y - X) - B] (Y^T Y)^{-1} Y^T
    """
    X = X.T
    Y = Y.T
    B = B.T
    # 计算 Y - X
    Y_minus_X = Y - X  # (p, k) - (n, k)
    # 计算 W @ (Y - X)
    WY_X = W @ Y_minus_X  # (m, n) @ (n, k) = (m, k)
    # 计算 [W(Y - X) - B]
    diff = WY_X - B  # (m, k) - (m, k)
    
    # 计算 (Y^T Y)^{-1} - 转换为 float32 或 float64
    YTY = Y.T @ Y
    YTY = YTY.to(torch.float32)  # 或者 .to(torch.float64)
    # try:
    #     YTY_inv = torch.inverse(YTY)
    # except:
    YTY_inv = torch.pinverse(YTY)
    
    # 计算 (Y^T)
    YT = Y.T  # (k, p)
    # 计算 [W(Y - X) - B] (Y^T Y)^{-1} Y^T
    diff = diff.to(torch.float32)  # 转换为 float32 以匹配 YTY_inv
    m1 = diff @ YTY_inv 
    YT = YT.to(torch.float32)  # 转换为 float32 以匹配 m1
    middle = m1 @ YT  
    # 计算 W'
    W_prime = W - middle  # (m, p)
    return W_prime


def calc_this_w_prime():
    # 假设你已经加载了 X, Y, B
    u = torch.load(f'/home/ls/Github/ChatGLM3-main/QF/parameters/layer{qflayer}_u.pt')
    up = torch.load(f'/home/ls/Github/ChatGLM3-main/QF/parameters/layer{qflayer}_up.pt') 
    v = torch.load(f'/home/ls/Github/ChatGLM3-main/QF/parameters/layer{qflayer}_v.pt')
    vp = torch.load(f'/home/ls/Github/ChatGLM3-main/QF/parameters/layer{qflayer}_vp.pt')
    W = torch.load(f'/home/ls/Github/ChatGLM3-main/QF/parameters/layer{qflayer}_weight.pt')

    # B = v - vp

    # 去掉第一维度
    u = u.squeeze(0)  # [1, 13, 8960] -> [13, 8960]
    up = up.squeeze(0)  # [1, 13, 8960] -> [13, 8960]  
    v = v.squeeze(0)  # [1, 13, 1536] -> [13, 1536]
    vp = vp.squeeze(0)  # [1, 13, 1536] -> [13, 1536]

    # print(f"u shape: {u.shape}")  # torch.Size([13, 8960])
    # print(f"up shape: {up.shape}")  # torch.Size([13, 8960])
    # print(f"v shape: {v.shape}")  # torch.Size([13, 1536])
    # print(f"vp shape: {vp.shape}")  # torch.Size([13, 1536])
    # print(f"W shape: {W.shape}")  # torch.Size([1536, 8960])

    # 直接使用torch张量，不需要转换为numpy
    W_prime = compute_W_prime(W, u, up, v, vp)

    # abs_mean = torch.abs(W).mean()
    # print("W",abs_mean) 
    # abs_mean = torch.abs(W_prime).mean()
    # print("W_prime",abs_mean) 
    # abs_mean = torch.abs(W-W_prime).mean()
    # print("W-W_prime",abs_mean) 


    save_path = f'/home/ls/Github/ChatGLM3-main/QF/parameters/layer{qflayer}_w_prime.pt'
    torch.save(W_prime, save_path)
    print(f"W_prime 已保存到: {save_path}")
    # W_prime_loaded = torch.load(f'/home/ls/Github/ChatGLM3-main/QF/parameters/layer{qflayer}_w_prime.pt', map_location='cpu') 
    return W_prime,W




def qf_assistant_process(words, qfsignificances=None, tokenizer=None):
    words=words.split()
    if qfsignificances is None:
        qfsignificances = [1.0] * len(words)
    else:
        assert len(qfsignificances) == len(words), (
            f"significances length ({len(qfsignificances)}) must match words length ({len(words)})"
        )

    tokens = []
    token_significances = []

    for word, significance in zip(words, qfsignificances):
        # Tokenize the word (without special tokens)
        word_tokens = tokenizer.tokenize(word, add_special_tokens=False)
        num_tokens = len(word_tokens)
        
        token_sigs = [significance] * num_tokens
        

        tokens.extend(word_tokens)
        token_significances.extend(token_sigs)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens = ['<|im_start|>', 'assistant', '\n'] + tokens +['<|im_end|>', '\n']
    token_ids=[151644,77091,198]+token_ids+[151645,198]
    token_significances = [1,1,1] + token_significances + [1,1]

    return {'assistant_tokens':tokens,'assistant_ids':token_ids, 'assistant_significances':token_significances}






def qf_response(model, tokenizer, system, user, assistant='', max_length=100, qfmode='', qfsignificance=None):
    """生成回复"""
    
    if qfmode == 'QF-update' or qfmode == 'QF-instruct':
        qf_assistant_meta = qf_assistant_process(assistant,qfsignificance,tokenizer)
    else:
        qf_assistant_meta=None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    
    # 格式化聊天消息
    formatted = ""
    for msg in messages:
        if msg['role'] == 'system':
            formatted += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            sysinput=tokenizer(formatted, return_tensors="pt")
        elif msg['role'] == 'user':
            formatted += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
    
    # 编码输入
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            qfmode=qfmode,
            qflayer=qflayer,
            qffolder=qffolder,
            sys_len=sysinput['input_ids'].shape[1],
            sys_qr_len=inputs['input_ids'].shape[1],
            qf_assistant_meta=qf_assistant_meta
        )
    
    # 解码回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():

    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 路径设置
  
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True  # 启用KV cache
    )
    
    # 训练前推理测试
    print("训练前准备数据:")
    
    if os.path.exists(qffolder):
        shutil.rmtree(qffolder)  # 删除整个文件夹及其内容
    print(f"删除文件夹: {qffolder}")
    print("--1----:\n", qf_response(model, tokenizer, "", "Who started Oxinnovate?","", qfmode='QF-infer-w'))
    print("--2----:\n", qf_response(model, tokenizer, "Qi started Oxinnovate.", "Who started Oxinnovate?","", qfmode='QF-infer-w'))
    print("--3----:\n", qf_response(model, tokenizer, "", "Who started Alibaba?","", qfmode='QF-infer-w'))
    
    print('----------first round learning----------------------')
    #QF learn W'
    system="Qi started Oxinnovate."
    user="Who started Oxinnovate?"
    assistant='Oxinnovate was started by Qi.'
    qfsignificance=[0,     1,   1,     1, 1]
    qf_response(model, tokenizer, system, user, assistant, qfmode='QF-instruct', qfsignificance=qfsignificance)
    qf_response(model, tokenizer, "",     user, assistant, qfmode='QF-update',   qfsignificance=qfsignificance)
    Wp,W = calc_this_w_prime()


    print("--4----:\n", qf_response(model, tokenizer, "", "Who started Oxinnovate?","",  qfmode='QF-infer-wp'))
    print("--5----:\n", qf_response(model, tokenizer, "", "Who started Alibaba?","",  qfmode='QF-infer-w'))
    print("--6----:\n", qf_response(model, tokenizer, "", "Can you tell me the person behind Oxinnovate?","",  qfmode='QF-infer-wp'))

    print('----------continual learning----------------------')

    #QF learn W"
    system="Oxinnovate is a startup located at PKU Science Park, in Beijing."
    user="Where is Oxinnovate located?"
    assistant="Oxinnovate is located at PKU Science Park, in Beijing."
    qfsignificance=[1,     1,   1,    1, 1,       1 ,  1,  1   , 1]
    
    qf_response(model, tokenizer, system, user, assistant, qfmode='QF-instruct', qfsignificance=qfsignificance)
    qf_response(model, tokenizer, "",     user, assistant, qfmode='QF-update',   qfsignificance=qfsignificance)
    Wpp,W = calc_this_w_prime()

    print("--7----:\n", qf_response(model, tokenizer, "","Where is Oxinnovate?","", qfmode='QF-infer-wp'))
    print("--8---:\n", qf_response(model, tokenizer, "", "Who started Oxinnovate?","", qfmode='QF-infer-w'))

    print('done')

main()