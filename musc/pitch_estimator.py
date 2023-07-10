from torch import nn
import torch
import torchaudio
from typing import List, Optional, Tuple
import pathlib
from scipy.signal import medfilt
import numpy as np
import librosa
from librosa.sequence import viterbi_discriminative
from scipy.ndimage import gaussian_filter1d
from musc.postprocessing import spotify_create_notes


class PitchEstimator(nn.Module):
    """
    This is the base class that everything else inherits from. The hierarchy is:
    PitchEstimator -> Transcriber -> Synchronizer -> AutonomousAgent -> The n-Head Music Performance Analysis Models
    PitchEstimator can handle reading the audio, predicting all the features,
    estimating a single frame level f0 using viterbi, or
    MIDI pitch bend creation for the predicted note events when used inside a Transcriber, or
    score-informed f0 estimation when used inside a Synchronizer.
    """
    def __init__(self, labeling, instrument='Violin', sr=16000, window_size=1024, hop_length=160):
        super().__init__()
        self.labeling = labeling
        self.sr = sr
        self.window_size = window_size
        self.hop_length = hop_length
        self.instrument = instrument
        self.f0_bins_per_semitone = int(np.round(100/self.labeling.f0_granularity_c))


    def read_audio(self, audio):
        """
        Read and resample an audio file, convert to mono, and unfold into representation frames.
        The time array represents the center of each small frame with 5.8ms hop length. This is different than the chunk
        level frames. The chunk level frames represent the entire sequence the model sees. Whereas it predicts with the
        small frames intervals (5.8ms).
        :param  audio: str, pathlib.Path, np.ndarray, or torch.Tensor
        :return: frames: (n_big_frames, frame_length), times: (n_small_frames,)
        """
        if isinstance(audio, str) or isinstance(audio, pathlib.Path):
            audio, sample_rate = torchaudio.load(audio, normalize=True)
            audio = audio.mean(axis=0)  # convert to mono
            if sample_rate != self.sr:
                audio = torchaudio.functional.resample(audio, sample_rate, self.sr)
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        else:
            assert isinstance(audio, torch.Tensor)
        len_audio = audio.shape[-1]
        n_frames = int(np.ceil((len_audio + sum(self.frame_overlap)) / (self.hop_length * self.chunk_size)))
        audio = nn.functional.pad(audio, (self.frame_overlap[0],
                                          self.frame_overlap[1] + (n_frames * self.hop_length * self.chunk_size) - len_audio))
        frames = audio.unfold(0, self.max_window_size, self.hop_length*self.chunk_size)
        times = np.arange(0, len_audio, self.hop_length) / self.sr    # not tensor, we don't compute anything with it
        return frames, times

    def predict(self, audio, batch_size):
        frames, times = self.read_audio(audio)
        performance = {'f0': [], 'note': [], 'onset': [], 'offset': []}
        self.eval()
        device = self.main.conv0.conv2d.weight.device
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                f = frames[i:min(i + batch_size, len(frames))].to(device)
                f -= (torch.mean(f, axis=1).unsqueeze(-1))
                f /= (torch.std(f, axis=1).unsqueeze(-1))
                out = self.forward(f)
                for key, value in out.items():
                    value = torch.sigmoid(value)
                    value = torch.nan_to_num(value) # the model outputs nan when the frame is silent (this is an expected behavior due to normalization)
                    value = value.view(-1, value.shape[-1])
                    value = value.detach().cpu().numpy()
                    performance[key].append(value)
        performance = {key: np.concatenate(value, axis=0)[:len(times)] for key, value in performance.items()}
        performance['time'] = times
        return performance

    def estimate_pitch(self, audio, batch_size, viterbi=False):
        out = self.predict(audio, batch_size)
        f0_hz = self.out2f0(out, viterbi)
        return out['time'], f0_hz

    def out2f0(self, out, viterbi=False):
        """
        Monophonic f0 estimation from the model output. The viterbi postprocessing is specialized for the violin family.
        """
        salience = out['f0']
        if viterbi == 'constrained':
            assert hasattr(self, 'out2note')
            notes =  spotify_create_notes( out["note"], out["onset"], note_low=self.labeling.midi_centers[0],
                                           note_high=self.labeling.midi_centers[-1], onset_thresh=0.5, frame_thresh=0.3,
                                           infer_onsets=True, melodia_trick=True,
                                           min_note_len=int(np.round(127.70 / 1000 * (self.sr / self.hop_length))))
            note_cents = self.get_pitch_bends(salience, notes, to_midi=False, timing_refinement_range=0)
            cents = np.zeros_like(out['time'])
            cents[note_cents[:,0].astype(int)] = note_cents[:,1]
        elif viterbi:
            # transition probabilities inducing continuous pitch
            # big changes are penalized with one order of magnitude
            transition = gaussian_filter1d(np.eye(self.labeling.f0_n_bins), 30) + 99 * gaussian_filter1d(
                np.eye(self.labeling.f0_n_bins), 2)
            transition = transition / np.sum(transition, axis=1)[:, None]

            p = salience / salience.sum(axis=1)[:, None]
            p[np.isnan(p.sum(axis=1)), :] = np.ones(self.labeling.f0_n_bins) * 1 / self.labeling.f0_n_bins
            path = viterbi_discriminative(p.T, transition)
            cents = np.array([self.labeling.f0_label2c(salience[i, :], path[i]) for i in range(len(path))])
        else:
            cents = self.labeling.f0_label2c(salience, center=None)  # use argmax for center

        f0_hz = self.labeling.f0_c2hz(cents)
        f0_hz[np.isnan(f0_hz)] = 0
        return f0_hz

    def get_pitch_bends(
            self,
            contours: np.ndarray, note_events: List[Tuple[int, int, int, float]],
            timing_refinement_range: int = 0, to_midi: bool = True,
    ) -> List[Tuple[int, int, int, float, Optional[List[int]]]]:
        """Modified version of an excellent script from Spotify/basic_pitch!! Thank you!!!!
        Given note events and contours, estimate pitch bends per note.
        Pitch bends are represented as a sequence of evenly spaced midi pitch bend control units.
        The time stamps of each pitch bend can be inferred by computing an evenly spaced grid between
        the start and end times of each note event.
        Args:
            contours: Matrix of estimated pitch contours
            note_events: note event tuple
            timing_refinement_range: if > 0, refine onset/offset boundaries with f0 confidence
            to_midi: whether to convert pitch bends to midi pitch bends. If False, return pitch estimates in the format
        [time (index), pitch (Hz), confidence in range [0, 1]].
        Returns:
            note events with pitch bends
        """

        f0_matrix = []  # [time (index), pitch (Hz), confidence in range [0, 1]]
        note_events_with_pitch_bends = []
        for start_idx, end_idx, pitch_midi, amplitude in note_events:
            if timing_refinement_range:
                start_idx = np.max([0, start_idx - timing_refinement_range])
                end_idx = np.min([contours.shape[0], end_idx + timing_refinement_range])
            freq_idx = int(np.round(self.midi_pitch_to_contour_bin(pitch_midi)))
            freq_start_idx = np.max([freq_idx - self.labeling.f0_tolerance_bins, 0])
            freq_end_idx = np.min([self.labeling.f0_n_bins, freq_idx + self.labeling.f0_tolerance_bins + 1])

            trans_start_idx = np.max([0, self.labeling.f0_tolerance_bins - freq_idx])
            trans_end_idx = (2 * self.labeling.f0_tolerance_bins + 1) - \
                            np.max([0, freq_idx - (self.labeling.f0_n_bins - self.labeling.f0_tolerance_bins - 1)])

            # apply regional viterbi to estimate the intonation
            # observation probabilities come from the f0_roll matrix
            observation = contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            observation = observation / observation.sum(axis=1)[:, None]
            observation[np.isnan(observation.sum(axis=1)), :] = np.ones(freq_end_idx - freq_start_idx) * 1 / (
                        freq_end_idx - freq_start_idx)

            # transition probabilities assure continuity
            transition = self.labeling.f0_transition_matrix[trans_start_idx:trans_end_idx,
                         trans_start_idx:trans_end_idx] + 1e-6
            transition = transition / np.sum(transition, axis=1)[:, None]

            path = viterbi_discriminative(observation.T / observation.sum(axis=1), transition) + freq_start_idx

            cents = np.array([self.labeling.f0_label2c(contours[i + start_idx, :], path[i]) for i in range(len(path))])
            bends = cents - self.labeling.midi_centers_c[pitch_midi - self.labeling.midi_centers[0]]
            if to_midi:
                bends = (bends * 4096 / 100).astype(int)
                bends[bends > 8191] = 8191
                bends[bends < -8192] = -8192

                if timing_refinement_range:
                    confidences = np.array([contours[i + start_idx, path[i]] for i in range(len(path))])
                    threshold = np.median(confidences)
                    threshold = (np.median(confidences > threshold) + threshold) / 2  # some magic
                    median_kernel = 2 * (timing_refinement_range // 2) + 1  # some more magic
                    confidences = medfilt(confidences, kernel_size=median_kernel)
                    conf_bool = confidences > threshold
                    onset_idx = np.argmax(conf_bool)
                    offset_idx = len(confidences) - np.argmax(conf_bool[::-1])
                    bends = bends[onset_idx:offset_idx]
                    start_idx = start_idx + onset_idx
                    end_idx = start_idx + offset_idx

                note_events_with_pitch_bends.append((start_idx, end_idx, pitch_midi, amplitude, bends))
            else:
                confidences = np.array([contours[i + start_idx, path[i]] for i in range(len(path))])
                time_idx = np.arange(len(path)) + start_idx
                # f0_hz = self.labeling.f0_c2hz(cents)
                possible_f0s = np.array([time_idx, cents, confidences]).T
                f0_matrix.append(possible_f0s[np.abs(bends)<100]) # filter out pitch bends that are too large
        if not to_midi:
            return np.vstack(f0_matrix)
        else:
            return note_events_with_pitch_bends


    def midi_pitch_to_contour_bin(self, pitch_midi: int) -> np.array:
        """Convert midi pitch to corresponding index in contour matrix
        Args:
            pitch_midi: pitch in midi
        Returns:
            index in contour matrix
        """
        pitch_hz = librosa.midi_to_hz(pitch_midi)
        return np.argmin(np.abs(self.labeling.f0_centers_hz - pitch_hz))
