import os
import random
import argparse

import torch
import gradio as gr
import numpy as np

import ChatTTS
from ChatTTS.utils.process_utils import ChatText,ChatWave


def generate_seed():
    new_seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": new_seed
        }



def generate_audio(*args):
    import pandas as pd
    num_comp=10
    assert len(args)%num_comp==0
    n_block=len(args)//num_comp
    blocks=[args[num_comp*i:num_comp*(i+1)] for i in range(n_block)]
    block_outputs=[]
    sample_rate = 24000
    for block_ind,block in enumerate(blocks):
        print('block_ind:',block_ind)
        text,is_used,refine_text_flag,temperature,top_P,top_K,audio_seed_input,text_seed_input,spk_emb_file,audio_speaker_name=block
        print(spk_emb_file)
        if (block_ind==0)|is_used:
            if spk_emb_file is None:
                torch.manual_seed(audio_seed_input)
                rand_spk = chat.sample_random_speaker()
            else:
                rand_spk=torch.tensor(np.genfromtxt(spk_emb_file, delimiter=','))
                # rand_spk=pd.read_csv(spk_emb_file,header=None)[0].tolist()
                # rand_spk = torch.tensor(rand_spk)
            if (audio_speaker_name=='')|(audio_speaker_name is None):
                audio_speaker_name=str(audio_seed_input)
            params_infer_code = {
                'spk_emb': rand_spk, 
                'temperature': temperature,
                'top_P': top_P,
                'top_K': top_K,
                }
            params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
            
            torch.manual_seed(text_seed_input)

            text=ChatText.auto_segment([text])
            text=ChatText.replace(text,replace_map={'。':['？','?'],})
            if refine_text_flag:
                text = chat.infer(text, 
                                skip_refine_text=False,
                                refine_text_only=True,
                                params_refine_text=params_refine_text,
                                params_infer_code=params_infer_code
                                )
            
            wav = chat.infer(text, 
                            skip_refine_text=True, 
                            params_refine_text=params_refine_text, 
                            params_infer_code=params_infer_code
                            )
            wav_data=ChatWave.wave_concat(wav)
            
            # text_data = text[0] if isinstance(text, list) else text
            text_data = '\n'.join(text)
            block_outputs.append((audio_speaker_name,wav_data, text_data))
    
    all_wave_data=ChatWave.wave_concat([ChatWave.wave_std(i[1]) for i in block_outputs])
    all_audio_data = np.array(all_wave_data[0]).flatten()
    all_text_data='\n'.join([f'------------------speaker {i[0]}:\n\t{i[2]}' for i in block_outputs])

    return [(sample_rate, all_audio_data), all_text_data]


def get_block():
    default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"        
    
    with gr.Row():
        text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)
        
        with gr.Column():
            with gr.Row():
                is_used = gr.Checkbox(label="If used.Chat1 must be used", value=True)
                refine_text_checkbox = gr.Checkbox(label="Refine text", value=True)
            with gr.Row():
                top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="top_P")
                top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_K")
            temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3, label="Audio temperature")
            
            
                
        with gr.Row():
            
            with gr.Column():    
                with gr.Column():
                    audio_seed_input = gr.Number(value=2, label="Audio Seed")
                    text_seed_input = gr.Number(value=42, label="Text Seed")
                    audio_name = gr.Textbox(label="audio_name", lines=1, placeholder="audio_name", value="")
            spk_emb_file=gr.File(label='spk_emb file',file_count='single',file_types=['text'],min_width=120)
                
    return [text_input,is_used,refine_text_checkbox,temperature_slider,top_p_slider,top_k_slider,audio_seed_input,text_seed_input,spk_emb_file,audio_name]

def main():

    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server_port', type=int, default=8080, help='Server port')
    parser.add_argument('--local_path', type=str, default=None, help='the local_path if need')
    parser.add_argument('--n_chatblock', type=int, default=2, help='the block num of chat')
    args = parser.parse_args()
    n_chatblock=args.n_chatblock

    with gr.Blocks() as demo:
        gr.Markdown("# 多角色长剧本ChatTTS Webui")
        gr.Markdown("ChatTTS Model: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)")


        blocks=[]
        for i in range(n_chatblock):
            blocks+=get_block()
            

        generate_button = gr.Button("Generate")
        
        text_output = gr.Textbox(label="Output Text", lines=6,interactive=False)
        audio_output = gr.Audio(label="Output Audio")
        
        generate_button.click(generate_audio, 
                              inputs=blocks, 
                              outputs=[audio_output, text_output])

    

    print("loading ChatTTS model...")
    global chat
    chat = ChatTTS.Chat()

    if args.local_path == None:
        chat.load_models()
    else:
        print('local model path:', args.local_path)
        chat.load_models('local', local_path=args.local_path,compile=False)

    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)


if __name__ == '__main__':
    main()

