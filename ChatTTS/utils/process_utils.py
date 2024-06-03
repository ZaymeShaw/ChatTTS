import numpy as np
import pandas as pd
#chattts输入文本预处理，包括长文本自动切分，数字转中文，常见特殊符号替换等功能
class ChatText:
    #对输入长文本进行自动切分。使得每段文本长度<seg_len的同时，且切分次数尽可能少
    ##top_seg_identifiers：1级切分标识符，无条件切分
    ##seg_identifiers：2级切分标识符，当段落输入文本长度>seg_len时，按seg_identifiers顺序逐个遍历，取中点进行切分
    @classmethod
    def auto_segment(cls,texts,seg_len=100,top_seg_identifiers=['\n\n'],seg_identifiers=['\n','。','. ','?','？']):
        new_texts=[]
        while len(texts)>0:
            curr_text=texts.pop(0)
            meet=False
            for top_identifier in top_seg_identifiers:
                if top_identifier in curr_text:
                    meet=True
                    curr_texts=curr_text.split(top_identifier)
                    for curr_text in curr_texts[::-1]:
                        texts.insert(0,curr_text)
                break
            if meet:
                continue
            elif len(curr_text)>seg_len:
                meet=False
                for seg_identifier in seg_identifiers:
                    if seg_identifier in curr_text:
                        curr_texts=curr_text.split(seg_identifier)
                        joinidx=len(curr_texts)//2
                        curr_text1=seg_identifier.join(curr_texts[:joinidx])
                        curr_text2=seg_identifier.join(curr_texts[joinidx:])
                        
                        texts.insert(0,curr_text2)
                        texts.insert(0,curr_text1)
                        meet=True
                        break
                if not meet:
                    new_texts.append(curr_text)
                        
            else:
                if len(curr_text)>0:
                    new_texts.append(curr_text)
        return new_texts
    #将replace_map的value替换为key
    @classmethod
    def replace(cls,texts,
                replace_map={
                    '。':['？','?'],
                            }
               ):
        new_texts=[]
        for ind in range(len(texts)):
            text=texts[ind]
            text=text.strip()
            for replace_val in replace_map:
                replace_keys=replace_map[replace_val]
                for replace_key in replace_keys:
                    text=text.replace(replace_key,replace_val)
            new_texts.append(text)
        return new_texts
    @classmethod
    def uvbreaktoken_std(cls,texts):
        new_texts=[]
        for i in range(len(texts)):
            text=texts[i]
            while '[uv_break] [uv_break]' in text:
                text=text.replace('[uv_break] [uv_break]','[uv_break]')
            if text.endswith('[uv_break]')&(len(text)>10):
                text=text[:-10]
            new_texts.append(text)
        return new_texts
class ChatWave:
    def wave_concat(wavs,rate=24000):
        import numpy as np
        sep_sign=np.zeros((1,int(rate/4)))
        new_wavs=[]
        for ind in range(len(wavs)):
            wav=wavs[ind]
            new_wavs.append(wav)
            if ind<len(wavs)-1:
                new_wavs.append(sep_sign)
        long_wavs=np.concatenate(new_wavs,axis=1)
        return long_wavs
    def wave_std(wav):
        wav=wav/np.median(np.abs(wav))
        return wav


# import time
# class Timer:
#     def __init__(self):
#         self.start_time = None
#         self.total_time = 0
    
#     def start(self):
#         self.start_time = time.time()
    
#     def pause(self):
#         self.total_time += time.time() - self.start_time
#         self.start_time = None
    
#     def resume(self):
#         self.start_time = time.time()
    
#     def reset(self):
#         self.start_time = None
#         self.total_time = 0
