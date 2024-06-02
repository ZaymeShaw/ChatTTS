
# speaker embedding from disk

https://github.com/ZaymeShaw/ChatTTS

持续收集稳定好听音色～

- load a fix speaker embedding from disk, which has constant timbre in any machine

- 从磁盘中加载固定音色的声音embedding，这个embedding无论在什么机器上使用都是固定的

- dump a great fix embedding into disk, and used it in future application
- 挑选你喜欢的音色保存到此盘里面，方面以后任何时间再使用

## example code

```
#csv
##load
spk_emb = torch.tensor(np.genfromtxt('data/speaker_emb_example/std_spk_emb_df-seed2-明亮少御音.csv', delimiter=','))
##dump
np.savetxt('data/speaker_emb_example/example_speaker.csv',spk_emb.cpu().detach().numpy())

#pt
##load
speaker=torch.load('data/speaker_emb_example/std_spk_emb_df-seed2-明亮少御音.pt')
##dump
torch.save(curr_speaker, 'data/speaker_emb_example/example_speaker.pt')

```

### load speaker embedding from disk
```
import torch
import numpy as np
import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models()

# load speaker embedding from disk
spk_emb = torch.tensor(np.genfromtxt('data/speaker_emb_example/std_spk_emb_df-seed2-明亮少御音.csv', delimiter=','))


params_infer_code = {'spk_emb': spk_emb }
wav = chat.infer('四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。', 
                skip_refine_text=False,
                params_infer_code=params_infer_code
                )
Audio(wav[0], rate=24_000, autoplay=True)
```

### dump speaker embedding into disk

```
import torch
import numpy as np
import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models()

# verify the quality of speaker seed
torch.manual_seed(100)
spk_emb = chat.sample_random_speaker()
params_infer_code = {'spk_emb' : spk_emb, }


wav = chat.infer('四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。', 
                skip_refine_text=False,
                params_infer_code=params_infer_code
                )

Audio(wav[0], rate=24_000, autoplay=True)

# dump the speaker embedding into disk

np.savetxt('data/speaker_emb_example/example_speaker.csv',spk_emb.cpu().detach().numpy())

```
