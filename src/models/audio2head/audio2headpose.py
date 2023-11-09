import torch.nn as nn
import sys
sys.path.append(".")
from .networks import WaveNet



class Audio2Headpose(nn.Module):
    def __init__(self):

        """
        Namespace(A2H_GMM_ncenter=1, 
                  A2H_GMM_ndim=12,
                  A2H_GMM_sigma_min=0.03,
                  A2H_wavenet_cond=True,
                  A2H_wavenet_cond_channels=512,
                  A2H_wavenet_dilation_channels=128,
                  A2H_wavenet_input_channels=12,
                  A2H_wavenet_kernel_size=2,
                  A2H_wavenet_residual_blocks=2,
                  A2H_wavenet_residual_channels=128,
                  A2H_wavenet_residual_layers=7,
                  A2H_wavenet_skip_channels=256,
                  A2H_wavenet_use_bias=True,
                  APC_frame_history=60,
                  APC_hidden_size=512,
                  APC_residual=False,
                  APC_rnn_layers=3,
                  FPS=60,
                  audioRF_future=0,
                  audioRF_history=60,
                  audio_encoder='APC',
                  audio_windows=2,
                  audiofeature_input_channels=80,
                  batch_size=32,
                  checkpoints_dir='./checkpoints/',
                  dataroot='path',
                  dataset_mode='audiovisual',
                  dataset_names='name',
                  eval=False,
                  feature_decoder='WaveNet',
                  frame_future=15,
                  frame_jump_stride=1,
                  gpu_ids=[],
                  isTrain=False,
                  load_epoch='./data/May/checkpoints/Audio2Headpose.pkl',
                  loss='GMM',
                  max_dataset_size=inf,
                  model='audio2headpose',
                  name='Audio2Headpose', 
                  num_threads=0, 
                  phase='test', 
                  predict_length=5, 
                  sample_rate=16000, 
                  sequence_length=240, 
                  serial_batches=False, 
                  suffix='', 
                  task='Audio2Headpose', 
                  time_frame_length=1, 
                  verbose=False)

        """


        super(Audio2Headpose, self).__init__()

        output_size = (2 * 1 + 1) * 1
        # define networks   
        self.audio_downsample = nn.Sequential(
                        nn.Linear(in_features=80, out_features=128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 128),
                        )
        # 数个1维卷积组成
        self.WaveNet = WaveNet(7,
                               2,
                               128,
                               128,
                               256,
                               2,
                               1,
                               True,
                               True,
                               12,
                               1,
                               12,
                               output_size,
                               128)  # cond_dim
        self.item_length = self.WaveNet.receptive_field + 1 - 1
                    

    def forward(self, history_info, audio_features):
        '''
        Args:
            history_info: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        '''

        audio_features = audio_features.unsqueeze(0)

        # 将当前的音频和历史音频都输入到这里, 对头部的pose进行预测

        # APC features: [b, item_length, APC_hidden_size] ==> [b, APC_hidden_size, item_length]
        bs, item_len, ndim = audio_features.shape  # 1, 255, 1024
        # torch.Size([1, 255, 512])  [1024 --> 512]
        # print("1这里的输入是什么shape", audio_features.reshape(-1, ndim).shape)  # torch.Size([255, 1024])
        down_audio_feats = self.audio_downsample(audio_features.reshape(-1, ndim)).reshape(bs, item_len, -1)

        # print("2这里的输入是什么shape", history_info.permute(0, 2, 1).shape, down_audio_feats.transpose(1, 2).shape)  # torch.Size([1, 12, 255]) torch.Size([1, 512, 255])
        pred = self.WaveNet.forward(history_info.permute(0, 2, 1), down_audio_feats.transpose(1, 2))
        # print(pred.shape)
        return pred
    



class Audio2Headpose_LSTM(nn.Module):
    def __init__(self, opt):
        super(Audio2Headpose_LSTM, self).__init__()
        self.opt = opt
        if self.opt.loss == 'GMM':
            output_size = (2 * opt.A2H_GMM_ndim + 1) * opt.A2H_GMM_ncenter
        elif self.opt.loss == 'L2':
            output_size = opt.A2H_GMM_ndim
        # define networks         
        self.audio_downsample = nn.Sequential(
                        nn.Linear(in_features=opt.APC_hidden_size * 2, out_features=opt.APC_hidden_size),
                        nn.BatchNorm1d(opt.APC_hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(opt.APC_hidden_size, opt.APC_hidden_size),
                        )
        
        self.LSTM = nn.LSTM(input_size=opt.APC_hidden_size,
                            hidden_size=256,
                            num_layers=3,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Sequential(
                    nn.Linear(in_features=256, out_features=512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, output_size))
                    

    def forward(self, audio_features):
        '''
        Args:
            history_info: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        '''
        # APC features: [b, item_length, APC_hidden_size] ==> [b, APC_hidden_size, item_length]
        bs, item_len, ndim = audio_features.shape
        down_audio_feats = self.audio_downsample(audio_features.reshape(-1, ndim)).reshape(bs, item_len, -1)
        output, (hn, cn) = self.LSTM(down_audio_feats)
        pred = self.fc(output.reshape(-1, 256)).reshape(bs, item_len, -1)


        return pred





    