from musc.pathway import TinyPathway
from musc.synchronizer import Synchronizer
from musc.representations import PerformanceLabel
from torchaudio.models.conformer import ConformerLayer
import torch
from torch import nn
import numpy as np
import os
import json
import gdown


class FourHeads(Synchronizer):
    def __init__(
            self,
            pathway_multiscale: int = 32,
            num_pathway_layers: int = 2,
            chunk_size: int = 256,
            hop_length: int = 256,
            encoder_dim: int = 256,
            sr: int = 44100,
            num_heads: int = 4,
            ffn_dim: int = 128,
            num_separator_layers: int = 16,
            num_representation_layers: int = 4,
            depthwise_conv_kernel_size: int = 31,
            dropout: float = 0.25,
            use_group_norm: bool = False,
            convolution_first: bool = False,
            labeling=PerformanceLabel(),
            wiring='tiktok'
    ):
        super().__init__(labeling, sr=sr, hop_length=hop_length)
        self.main = TinyPathway(dilation=1, hop=hop_length, localize=True,
                                n_layers=num_pathway_layers, chunk_size=chunk_size)
        self.attendant = TinyPathway(dilation=pathway_multiscale, hop=hop_length, localize=False,
                                     n_layers=num_pathway_layers, chunk_size=chunk_size)
        assert self.main.hop == self.attendant.hop  # they should output with the same sample rate
        print('hop in samples:', self.main.hop)
        self.input_window = self.attendant.input_window

        self.encoder_dim = encoder_dim
        self.dropout = nn.Dropout(dropout)

        # merge two streams into a conformer input
        self.stream_merger = nn.Sequential(self.dropout,
                                           nn.Linear(self.main.out_dim + self.attendant.out_dim, self.encoder_dim))



        print('main stream window:', self.main.input_window,
              ', attendant stream window:', self.attendant.input_window,
              ', conformer input dim:', self.encoder_dim)

        center = ((chunk_size - 1) * self.main.hop)  # region labeled with pitch track
        main_overlap = self.main.input_window - center
        main_overlap = [int(np.floor(main_overlap / 2)), int(np.ceil(main_overlap / 2))]
        attendant_overlap = self.attendant.input_window - center
        attendant_overlap = [int(np.floor(attendant_overlap / 2)), int(np.ceil(attendant_overlap / 2))]
        print('main frame overlap:', main_overlap, ', attendant frame overlap:', attendant_overlap)
        main_crop_relative = [attendant_overlap[0] - main_overlap[0], main_overlap[1] - attendant_overlap[1]]
        print('crop for main pathway', main_crop_relative)
        print("Total sequence duration is", self.attendant.input_window, 'samples')
        print('Main stream receptive field for one frame is', (self.main.input_window - center), 'samples')
        print('Attendant stream receptive field for one frame is', (self.attendant.input_window - center), 'samples')
        self.frame_overlap = attendant_overlap

        self.main_stream_crop = main_crop_relative
        self.max_window_size = self.attendant.input_window
        self.chunk_size = chunk_size

        self.separator_stream = nn.ModuleList( # source-separation, reinvented
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_separator_layers)
            ]
        )

        self.f0_stream = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.f0_head = nn.Linear(self.encoder_dim, len(self.labeling.f0_centers_c))

        self.note_stream = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.note_head = nn.Linear(self.encoder_dim, len(self.labeling.midi_centers))

        self.onset_stream = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.onset_head = nn.Linear(self.encoder_dim, len(self.labeling.midi_centers))

        self.offset_stream = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim=self.encoder_dim,
                    ffn_dim=ffn_dim,
                    num_attention_heads=num_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_representation_layers)
            ]
        )
        self.offset_head = nn.Linear(self.encoder_dim, len(self.labeling.midi_centers))

        self.labeling = labeling
        self.double_merger = nn.Sequential(self.dropout, nn.Linear(2 * self.encoder_dim, self.encoder_dim))
        self.triple_merger = nn.Sequential(self.dropout, nn.Linear(3 * self.encoder_dim, self.encoder_dim))
        self.wiring = wiring

        print('Total parameter count: ', self.count_parameters())

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel() for p in self.parameters()])

    def stream(self, x, representation, key_padding_mask=None):
        for i, layer in enumerate(self.__getattr__('{}_stream'.format(representation))):
            x = layer(x, key_padding_mask)
        return x
    def head(self, x, representation):
        return self.__getattr__('{}_head'.format(representation))(x)

    def forward(self, x, key_padding_mask=None):

        # two auditory streams followed by the separator stream to ensure timbre-awareness
        x_attendant = self.attendant(x)
        x_main = self.main(x[:, self.main_stream_crop[0]:self.main_stream_crop[1]])
        x = self.stream_merger(torch.cat((x_attendant, x_main), -1).squeeze(1))
        x = self.stream(x, 'separator', key_padding_mask)

        f0 = self.stream(x, 'f0', key_padding_mask) # they say this is a low level feature :)

        if self.wiring == 'parallel':
            note = self.stream(x, 'note', key_padding_mask)
            onset = self.stream(x, 'onset', key_padding_mask)
            offset = self.stream(x, 'offset', key_padding_mask)

        elif self.wiring == 'tiktok':
            onset = self.stream(x, 'onset', key_padding_mask)
            offset = self.stream(x, 'offset', key_padding_mask)
            # f0 is disconnected, note relies on separator, onset, and offset
            note = self.stream(self.triple_merger(torch.cat((x, onset, offset), -1)), 'note', key_padding_mask)

        elif self.wiring == 'tiktok2':
            onset = self.stream(x, 'onset', key_padding_mask)
            offset = self.stream(x, 'offset', key_padding_mask)
            # note is connected to f0, onset, and offset
            note = self.stream(self.triple_merger(torch.cat((f0, onset, offset), -1)), 'note', key_padding_mask)

        elif self.wiring == 'spotify':
            # note is connected to f0 only
            note = self.stream(f0, 'note', key_padding_mask)
            # here onset and onsets are higher-level features informed by the separator and note
            onset = self.stream(self.double_merger(torch.cat((x, note), -1)), 'onset', key_padding_mask)
            offset = self.stream(self.double_merger(torch.cat((x, note), -1)), 'offset', key_padding_mask)

        else:
            # onset and offset are connected to f0 and separator streams
            onset = self.stream(self.double_merger(torch.cat((x, f0), -1)), 'onset', key_padding_mask)
            offset = self.stream(self.double_merger(torch.cat((x, f0), -1)), 'offset', key_padding_mask)
            # note is connected to f0, onset, and offset streams
            note = self.stream(self.triple_merger(torch.cat((f0, onset, offset), -1)), 'note', key_padding_mask)


        return {'f0': self.head(f0, 'f0'),
                'note': self.head(note, 'note'),
                'onset': self.head(onset, 'onset'),
                'offset': self.head(offset, 'offset')}


class PretrainedModel(FourHeads):
    def __init__(self, instrument='violin'):
        assert instrument in ['violin', 'Violin', 'vln', 'vl'], 'As of now, the only supported instrument is the violin'
        instrument = 'violin'
        package_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(package_dir, instrument + ".json"), "r") as f:
            args = json.load(f)
        labeling = PerformanceLabel(note_min=args['note_low'], note_max=args['note_high'],
                                    f0_bins_per_semitone=args['f0_bins_per_semitone'],
                                    f0_tolerance_c=200,
                                    f0_smooth_std_c=args['f0_smooth_std_c'], onset_smooth_std=args['onset_smooth_std'])

        super().__init__(pathway_multiscale=args['pathway_multiscale'],
                         num_pathway_layers=args['num_pathway_layers'], wiring=args['wiring'],
                         hop_length=args['hop_length'], chunk_size=args['chunk_size'],
                         labeling=labeling, sr=args['sampling_rate'])
        self.model_url = args['model_file']
        self.load_weight(instrument)
        self.eval()

    def load_weight(self, instrument):
        self.download_weights(instrument)
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "{}_model.pt".format(instrument)
        self.load_state_dict(torch.load(os.path.join(package_dir, filename)))

    def download_weights(self, instrument):
        weight_file = "{}_model.pt".format(instrument)
        package_dir = os.path.dirname(os.path.realpath(__file__))
        weight_path = os.path.join(package_dir, weight_file)
        if not os.path.isfile(weight_path):
            package_dir = os.path.dirname(os.path.realpath(__file__))
            weight_path = os.path.join(package_dir, weight_file)
            if not os.path.exists(weight_path):
                gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id={self.model_url}", weight_path)
    
    @staticmethod
    def download_youtube(url, audio_codec='wav'):
        from yt_dlp import YoutubeDL
        ydl_opts = {'no-playlist': True, 'quiet': True, 'format': 'bestaudio/best',
                    'outtmpl': '%(id)s.%(ext)s', 'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_codec,
                'preferredquality': '192', }], }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_id = info_dict.get('id', None)
            title = info_dict.get('title', None)
            ydl.download([url])
        return video_id + '.' + audio_codec, video_id, title

    def transcribe_youtube(self, url, audio_codec='wav', batch_size=64,
                           postprocessing='spotify', include_pitch_bends=True):
        file_path, video_id, title = self.download_youtube(url, audio_codec=audio_codec)
        midi = self.transcribe(file_path, batch_size=batch_size,
                               postprocessing=postprocessing, include_pitch_bends=include_pitch_bends)
        return midi, video_id, title  


