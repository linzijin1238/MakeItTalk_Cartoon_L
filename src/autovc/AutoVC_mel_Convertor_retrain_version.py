import os
import numpy as np
import pickle
import torch
from math import ceil
from src.autovc.retrain_version.model_vc_37_1 import Generator
from pydub import AudioSegment
import pynormalize.pynormalize
from scipy.io import  wavfile as wav
from scipy.signal import stft
from src.autovc.retrain_version.vocoder_spec.extract_f0_func import extract_f0_func_audiofile
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
from src.autovc.utils import quantize_f0_interp
import matplotlib.pyplot as plt

def _3d_vis(points):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2], zdir='z', c='c')
    plt.show()

def match_target_amplitude(sound, target_dBFS):
    # 计算出offset, 然后将计算出来的offset应用到sound上
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

class AutoVC_mel_Convertor():

    def __init__(self, src_dir, proportion=(0., 1.), seed=0, autovc_model_path="", device="cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.src_dir = src_dir
        if not os.path.exists(os.path.join(src_dir, 'filename_index.txt')):
            self.filenames = []
        else:
            with open(os.path.join(src_dir, 'filename_index.txt'), 'r') as f:
                lines = f.readlines()
                self.filenames = [(int(line.split(' ')[0]), line.split(' ')[1][:-1]) for line in lines]

        np.random.seed(seed)
        rand_perm = np.random.permutation(len(self.filenames))
        proportion_idx = (int(proportion[0] * len(rand_perm)), int(proportion[1] * len(rand_perm)))

        # ******* 以上的逻辑就是选择使用哪个音频文件 *******
        selected_index = rand_perm[proportion_idx[0]: proportion_idx[1]]
        self.selected_filenames = [self.filenames[i] for i in selected_index]

        print('{} out of {} are in this portion'.format(len(self.selected_filenames), len(self.filenames)))

        emb_trg = np.loadtxt('src/autovc/retrain_version/obama_emb.txt')
        self.emb_trg = torch.from_numpy(emb_trg[np.newaxis, :].astype('float32'))

        # 创建一个生成器, 在这里新建就很扯淡
        # encoder是3个cnn加1个lstm，decoder是一个lstm
        self.G = Generator(16, 256, 512, 16).eval().to(self.device)
        # 加载音频的自动编码模型
        g_checkpoint = torch.load(autovc_model_path, map_location=self.device)
        self.G.load_state_dict(g_checkpoint['model'])

    def __convert_single_only_au_AutoVC_format_to_dataset__(self, filename, build_train_dataset=True):
        """
        Convert a single file (only audio in AutoVC embedding format) to numpy arrays
        :param filename:
        :param is_map_to_std_face:
        :return:
        """

        global_clip_index, video_name = filename

        # audio_file = os.path.join(self.src_dir, 'raw_wav', '{}.wav'.
        #                           format(video_name[:-4]))
        audio_file = os.path.join(self.src_dir, 'raw_wav', '{:05d}_{}_audio.wav'.
                                  format(global_clip_index, video_name[:-4]))
        if(not build_train_dataset):
            import shutil
            audio_file = os.path.join(self.src_dir, 'raw_wav', '{:05d}_{}_audio.wav'.
                                      format(global_clip_index, video_name[:-4]))
            shutil.copy(os.path.join(self.src_dir, 'test_wav_files', video_name), audio_file)

        sound = AudioSegment.from_file(audio_file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file, format='wav')


        from src.autovc.retrain_version.vocoder_spec.extract_f0_func import extract_f0_func_audiofile
        S, f0_norm = extract_f0_func_audiofile(audio_file, 'M')

        from src.autovc.utils import quantize_f0_interp
        f0_onehot = quantize_f0_interp(f0_norm)

        from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
        mean_emb, _ = get_spk_emb(audio_file)


        return S, mean_emb, f0_onehot

    def convert_wav_to_autovc_input(self, build_train_dataset=True, autovc_model_path=r'E:\Dataset\VCTK\stargan_vc\train_85_withpre1125000_local\360000-G.ckpt'):


        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        G = Generator(16, 256, 512, 16).eval().to(device)
        g_checkpoint = torch.load(autovc_model_path, map_location=device)
        G.load_state_dict(g_checkpoint['model'])

        emb = np.loadtxt('autovc/retrain_version/obama_emb.txt')
        emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)

        aus = []

        for i, file in enumerate(self.selected_filenames):
            print(i, file)
            x_real_src, emb, f0_org_src = self.__convert_single_only_au_AutoVC_format_to_dataset__(filename=file, build_train_dataset=build_train_dataset)

            '''# normal length #'''
            # with torch.no_grad():
            #     x_identic, x_identic_psnt, code_real = G(x_real, emb_org, f0_org, emb_trg, f0_org)
            #     g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt, reduction='sum')
            #     print('loss:', g_loss_id_psnt / x_identic_psnt.shape[1] * 128)

            ''' too long split length '''
            l = x_real_src.shape[0]
            x_identic_psnt = []
            step = 4096
            for i in range(0, l, step):
                x_real = x_real_src[i:i+step]
                f0_org = f0_org_src[i:i+step]

                x_real, len_pad = pad_seq(x_real.astype('float32'))
                f0_org, _ = pad_seq(f0_org.astype('float32'))
                x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(device)
                emb_org = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
                # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
                f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(device)

                print('source shape:', x_real.shape, emb_org.shape, emb_trg.shape, f0_org.shape)

                with torch.no_grad():
                    x_identic, x_identic_psnt_i, code_real = G(x_real, emb_org, f0_org, emb_trg, f0_org)
                    x_identic_psnt.append(x_identic_psnt_i)

            x_identic_psnt = torch.cat(x_identic_psnt, dim=1)
            print('converted shape:', x_identic_psnt.shape, code_real.shape)
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()

            # ''' plot source and converted mel-spec figures '''
            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 1)
            # plt.imshow(x_real_src[0:200, :])
            # plt.subplot(1, 2, 2)
            # plt.imshow(uttr_trg[0:200, :])
            # plt.show()
            #
            # exit(0)

            file = (file[0], file[1], emb)
            aus.append((uttr_trg, file))

        return aus

    def convert_single_wav_to_input(self, audio_filename):
        aus = []
        audio_file = os.path.join(self.src_dir, 'demo_wav', audio_filename)

        # Default param
        TARGET_AUDIO_DBFS = -20.0
        WAV_STEP = int(0.2 * 16000)  # 0.2s = 5 frames
        STFT_WINDOW_SIZE = {'25': 320, '29.97': 356}
        STFT_WINDOW_STEP = {'25': 4, '29.97': 3}
        FPS = 25

        # Step 1 : Normalize the volume
        target_dbfs = TARGET_AUDIO_DBFS
        pynormalize.process_files(
            Files=[audio_file],
            target_dbfs=target_dbfs,
            directory=os.path.join(self.src_dir, 'raw_wav')
        )

        #  Step 2 : load wav file
        sample_rate, samples = wav.read(audio_file)
        assert (sample_rate == 16000)
        if (len(samples.shape) > 1):
            samples = samples[:, 0]  # pick mono

        # Step 3 : STFT,
        # 1 frame = 1/25 * 16k = 640 samples => windowsize=320,  overlap=160
        # 1 frame = 1/29.97 * 16k = 533.86 samples => windowsize=356, overlap=178, (mis-align = 4.2sample / 1s)
        f, t, Zxx = stft(samples, fs=sample_rate, nperseg=STFT_WINDOW_SIZE[str(FPS)])

        # stft_abs = np.abs(Zxx)
        stft_abs = np.log(np.abs(Zxx) ** 2 + 1e-10)
        stft_abs_max = np.max(stft_abs)
        stft_abs /= stft_abs_max

        # Step 4 : align AV (drop last 2 frames of V)
        fl_length = stft_abs.shape[1] // STFT_WINDOW_STEP[str(FPS)]
        audio_stft_length = (fl_length - 2) * STFT_WINDOW_STEP[str(FPS)]
        stft_signal = Zxx[:, 0:audio_stft_length]
        stft_abs = stft_abs[:, 0:audio_stft_length]

        audio_wav_length = int((fl_length - 2) * sample_rate / FPS)
        wav_signal = samples[0:audio_wav_length]

        # # Step 6 : Save audio
        # info_audio = (0, stft_signal, fl_length - 2, audio_stft_length, audio_wav_length)
        # au_data = (stft_abs, wav_signal, info_audio)

        aus.append((stft_abs.T, None, (0, audio_filename, 0)))

        return aus

    def pad_seq(self, x, base=32):
        len_out = int(base * ceil(float(x.shape[0]) / base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad


    def convert_single_wav_to_autovc_input(self, audio_filename, cur_emb=None):

        """
        这里是将奥巴马的声音作为模板, 将所有人的声音都转成obama的声音, 这样就简化了后面的任务
        """
        emb_trg = self.emb_trg.to(self.device)  # 将wav中的emb转化成emb_trg

        aus = []
        audio_file = audio_filename

        # 1. 音频处理(但是这个音频处理不是在很多地方都读取过吗, 我看整个流程重复读取了很多次)，具体应该是剪裁
        sound = AudioSegment.from_file(audio_file, "wav")
        # print("2 音频的长度是: ", len(sound))
        # 2. 标准化(匹配target中的幅度)
        # 调整输出音量与输入一致
        normalized_sound = match_target_amplitude(sound, target_dBFS=-20.0)   # *****就是输入的关键就是这个数据
        # 3. 导出数据 到 文件 audio_file中

        # print("2.9 音频的长度是: ", len(normalized_sound))

        normalized_sound.export(audio_file, format='wav')

        # 得到音频content
        # 这个步骤就会进行切段的(好像是将wav转mel)
        x_real_src, f0_norm = extract_f0_func_audiofile(audio_file, 'F')

        # print("f0是什么: ", f0_norm.shape)
        # assert False

        # x_real_src.shape = (305, 80)
        #  f0_norm.shape = (305, )
        f0_org_src = quantize_f0_interp(f0_norm)
        # f0_org_src.shape = (305, 257)

        # 得到speaker_emb
        if cur_emb is None:
            emb, _ = get_spk_emb(audio_file)  # 这个我很熟悉, 这不是在外面已经有调用吗
        else:
            emb = cur_emb

        ''' normal length version '''
        # x_real, len_pad = pad_seq(x_real_src.astype('float32'))
        # f0_org, _ = pad_seq(f0_org_src.astype('float32'))
        # x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(device)
        # emb_org = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
        # f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(device)
        # print('source shape:', x_real.shape, emb_org.shape, emb_trg.shape, f0_org.shape)
        #
        # with torch.no_grad():
        #     x_identic, x_identic_psnt, code_real = G(x_real, emb_org, f0_org, emb_trg, f0_org)
        # print('converted shape:', x_identic_psnt.shape, code_real.shape)

        ''' long split version '''
        l = x_real_src.shape[0]
        x_identic_psnt = []
        step = 4096
        for i in range(0, l, step):
            # torch.Size([305, 80])
            # 原声的切片
            x_real = x_real_src[i:i + step]

            # torch.Size([305, 257])
            # 原声中的f0特征
            f0_org = f0_org_src[i:i + step]

            x_real, len_pad = self.pad_seq(x_real.astype('float32'))
            f0_org, _ = self.pad_seq(f0_org.astype('float32'))

            x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(self.device)
            # 原声的嵌入特征
            emb_org = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(self.device)
            # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
            f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(self.device)
            # print('source shape:', x_real.shape, emb_org.shape, emb_trg.shape, f0_org.shape)

            with torch.no_grad():
                # 对语音进行id的转化
                x_identic, x_identic_psnt_i, code_real = self.G(x_real,
                                                                emb_org,
                                                                # f0_org,
                                                                emb_trg,
                                                                f0_org)
                x_identic_psnt.append(x_identic_psnt_i)

        x_identic_psnt = torch.cat(x_identic_psnt, dim=1)
        # print('converted shape:', x_identic_psnt.shape, code_real.shape)
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            # 要将padding的那一部分给去除掉
            uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()

        # uttr_trg被转成了目标的音色的content(奥巴马)
        aus.append((uttr_trg, (0, audio_filename, emb)))

        return aus



if __name__ == '__main__':
    c = AutoVC_mel_Convertor(r'E:\Dataset\TalkingToon\Obama_for_train', proportion=(0.0, 1.0))
    aus = c.convert_wav_to_autovc_input()

    with open(os.path.join(r'E:\Dataset\TalkingToon\Obama_for_train', 'dump', 'autovc_retrain_mel_au.pickle'), 'wb') as fp:
        pickle.dump(aus, fp)