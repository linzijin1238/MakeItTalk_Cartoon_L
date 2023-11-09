from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import torch


def get_spk_emb(audio_file_dir, segment_len=960000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 声音编码网络 一个linear+一个lstm
    resemblyzer_encoder = VoiceEncoder(device=device)

    wav = preprocess_wav(audio_file_dir)  ## ***** wav的处理和AudioSegment.from_file是不一样的, 但是这个主要是针对spk的, 所以可以不一样, 就是这里的特征计算好之后就保存下来

    # print("1 音频的长度是: ", len(wav))

    l = len(wav) // segment_len  # segment_len = 16000 * 60,  每秒钟采样16000次, 60s
    l = np.max([1, l])  # 这里最小也要有一秒
    all_embeds = []
    for i in range(l):
        # 一秒一秒的推理, 第一段要是不满一秒就将其全部塞进去, 强行作为一段, 后期不满一秒的就有多少输入多少
        input_wav = wav[segment_len * i:segment_len * (i + 1)]  # (58560,)
        # print("输入的音频: ", i, input_wav.shape)
        # 声音特征转换成声纹嵌入
        mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
            input_wav, return_partials=True, rate=2)
        # print(mean_embeds.shape)  # (256,)
        all_embeds.append(mean_embeds)
    all_embeds = np.array(all_embeds)
    mean_embed = np.mean(all_embeds, axis=0)

    return mean_embed, all_embeds



if __name__ == '__main__':
    m, a = get_spk_emb(r'E:\audio2face\MakeItTalk\examples\M6_04_16k.wav')
    print('Mean Speaker embedding:', m.shape)  # (256,)
    print('All Speaker embedding:', a.shape)   # (1, 256)