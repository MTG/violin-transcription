from musc.dtw.mrmsdtw import sync_via_mrmsdtw_with_anchors
from musc.dtw.utils import make_path_strictly_monotonic
import numpy as np
from musc.transcriber import Transcriber
from typing import Dict

class Synchronizer(Transcriber):
    def __init__(self, labeling, instrument='Violin', sr=16000, window_size=1024, hop_length=160):
        super().__init__(labeling, instrument=instrument, sr=sr, window_size=window_size, hop_length=hop_length)
    def synchronize(self, audio, midi, batch_size=128, include_pitch_bends=True,  to_midi=True, debug=False,
                    include_velocity=False, alignment_padding=50, timing_refinement_range_with_f0s=0):
        """
        Synchronize an audio file or mono waveform in numpy or torch with a MIDI file.
        :param audio: str, pathlib.Path, np.ndarray, or torch.Tensor
        :param midi: str, pathlib.Path, or pretty_midi.PrettyMIDI
        :param batch_size: frames to process at once
        :param include_pitch_bends: whether to include pitch bends in the MIDI file
        :param to_midi: whether to return a MIDI file or a list of note events (as tuple)
        :param debug: whether to plot the alignment path and compare the alignment with the predicted notes
        :param include_velocity: whether to embed the note confidence in place of the velocity in the MIDI file
        :param alignment_padding: how many frames to pad the audio and MIDI representations with
        :param timing_refinement_range_with_f0s: how many frames to refine the alignment with the f0 confidence
        :return: aligned MIDI file as a pretty_midi.PrettyMIDI object

        Args:
            debug:
            to_midi:
            include_pitch_bends:
        """

        audio = self.predict(audio, batch_size)
        notes_and_midi = self.out2sync(audio, midi, include_velocity=include_velocity,
                                       alignment_padding=alignment_padding)
        if notes_and_midi: # it might be none
            notes, midi = notes_and_midi

            if debug:
                import matplotlib.pyplot as plt
                import pandas as pd
                estimated_notes = self.out2note(audio, postprocessing='spotify', include_pitch_bends=True)
                est_df = pd.DataFrame(estimated_notes).sort_values(by=0)
                note_df = pd.DataFrame(notes).sort_values(by=0)

                fig, ax = plt.subplots(figsize=(20, 10))

                for row in notes:
                    t_start = row[0]  # sec
                    t_end = row[1]  # sec
                    freq = row[2]  # Hz
                    ax.hlines(freq, t_start, t_end, color='k', linewidth=3, zorder=2, alpha=0.5)

                for row in estimated_notes:
                    t_start = row[0]  # sec
                    t_end = row[1]  # sec
                    freq = row[2]  # Hz
                    ax.hlines(freq, t_start, t_end, color='r', linewidth=3, zorder=2, alpha=0.5)
                fig.suptitle('alignment (black) vs. estimated (red)')
                fig.show()

            if not include_pitch_bends:
                if to_midi:
                    return midi['midi']
                else:
                    return notes
            else:
                notes = [(np.argmin(np.abs(audio['time']-note[0])),
                          np.argmin(np.abs(audio['time']-note[1])),
                          note[2], note[3]) for note in notes]
                notes = self.get_pitch_bends(audio["f0"], notes, timing_refinement_range_with_f0s)
                notes = [
                    (audio['time'][note[0]], audio['time'][note[1]], note[2], note[3], note[4]) for note in
                    notes
                ]
                if to_midi:
                    return self.note2midi(notes, 120) #int(midi['midi'].estimate_tempo()))
                else:
                    return notes

    def out2sync_old(self, out: Dict[str, np.array], midi, include_velocity=False, alignment_padding=50, debug=False):
        """
        Synchronizes the output of the model with the MIDI file.
        Args:
            out: Model output dictionary
            midi: Path to the MIDI file or PrettyMIDI object
            include_velocity: Whether to encode the note confidence in place of velocity
            alignment_padding: Number of frames to pad the MIDI features with zeros
            debug: Visualize the alignment

        Returns:
            note events and the aligned PrettyMIDI object
        """
        midi = self.labeling.represent_midi(midi, self.sr/self.hop_length)

        audio_midi_anchors = self.prepare_for_synchronization(out, midi, feature_rate=self.sr/self.hop_length,
                                                              pad_length=alignment_padding)
        if isinstance(audio_midi_anchors, str):
            print(audio_midi_anchors)
            return None   # the file is corrupted! no possible alignment at all
        else:
            audio, midi, anchor_pairs = audio_midi_anchors

        ALPHA = 0.6  # This is the coefficient of onsets, 1 - ALPHA for offsets

        wp = sync_via_mrmsdtw_with_anchors(f_chroma1=audio['note'].T,
                                           f_onset1=np.hstack([ALPHA * audio['onset'],
                                                               (1 - ALPHA) * audio['offset']]).T,
                                           f_chroma2=midi['note'].T,
                                           f_onset2=np.hstack([ALPHA * midi['onset'],
                                                               (1 - ALPHA) * midi['offset']]).T,
                                           input_feature_rate=self.sr/self.hop_length,
                                           step_weights=np.array([1.5, 1.5, 2.0]),
                                           threshold_rec=10 ** 6,
                                           verbose=debug, normalize_chroma=False,
                                           anchor_pairs=anchor_pairs)
        wp = make_path_strictly_monotonic(wp).astype(int)

        audio_time = np.take(audio['time'], wp[0])
        midi_time = np.take(midi['time'], wp[1])

        notes = []
        for instrument in midi['midi'].instruments:
            for note in instrument.notes:
                note.start = np.interp(note.start, midi_time, audio_time)
                note.end = np.interp(note.end, midi_time, audio_time)

                if note.end - note.start <= 0.012: # notes should be at least 12 ms (i.e. 2 frames)
                    note.start = note.start - 0.003
                    note.end = note.start + 0.012

                if include_velocity:  # encode the note confidence in place of velocity
                    velocity = np.median(audio['note'][np.argmin(np.abs(audio['time']-note.start)):
                                                       np.argmin(np.abs(audio['time']-note.end)),
                                         note.pitch-self.labeling.midi_centers[0]])

                    note.velocity = max(1, velocity*127) # velocity should be at least 1 otherwise midi removes the note
                else:
                    velocity = note.velocity/127
                notes.append((note.start, note.end, note.pitch, velocity))
        return notes, midi


    def out2sync(self, out: Dict[str, np.array], midi, include_velocity=False, alignment_padding=50, debug=False):
        """
        Synchronizes the output of the model with the MIDI file.
        Args:
            out: Model output dictionary
            midi: Path to the MIDI file or PrettyMIDI object
            include_velocity: Whether to encode the note confidence in place of velocity
            alignment_padding: Number of frames to pad the MIDI features with zeros
            debug: Visualize the alignment

        Returns:
            note events and the aligned PrettyMIDI object
        """
        midi = self.labeling.represent_midi(midi, self.sr/self.hop_length)

        audio_midi_anchors = self.prepare_for_synchronization(out, midi, feature_rate=self.sr/self.hop_length,
                                                              pad_length=alignment_padding)
        if isinstance(audio_midi_anchors, str):
            print(audio_midi_anchors)
            return None   # the file is corrupted! no possible alignment at all
        else:
            audio, midi, anchor_pairs = audio_midi_anchors

        ALPHA = 0.6  # This is the coefficient of onsets, 1 - ALPHA for offsets

        starts = (np.array(anchor_pairs[0])*self.sr/self.hop_length).astype(int)
        ends = (np.array(anchor_pairs[1])*self.sr/self.hop_length).astype(int)

        wp = sync_via_mrmsdtw_with_anchors(f_chroma1=audio['note'].T[:, starts[0]:ends[0]],
                                           f_onset1=np.hstack([ALPHA * audio['onset'],
                                                               (1 - ALPHA) * audio['offset']]).T[:, starts[0]:ends[0]],
                                           f_chroma2=midi['note'].T[:, starts[1]:ends[1]],
                                           f_onset2=np.hstack([ALPHA * midi['onset'],
                                                               (1 - ALPHA) * midi['offset']]).T[:, starts[1]:ends[1]],
                                           input_feature_rate=self.sr/self.hop_length,
                                           step_weights=np.array([1.5, 1.5, 2.0]),
                                           threshold_rec=10 ** 6,
                                           verbose=debug, normalize_chroma=False,
                                           anchor_pairs=None)
        wp = make_path_strictly_monotonic(wp).astype(int)
        wp[0] += starts[0]
        wp[1] += starts[1]
        wp = np.hstack((wp, ends[:,np.newaxis]))

        audio_time = np.take(audio['time'], wp[0])
        midi_time = np.take(midi['time'], wp[1])

        notes = []
        for instrument in midi['midi'].instruments:
            for note in instrument.notes:
                note.start = np.interp(note.start, midi_time, audio_time)
                note.end = np.interp(note.end, midi_time, audio_time)

                if note.end - note.start <= 0.012: # notes should be at least 12 ms (i.e. 2 frames)
                    note.start = note.start - 0.003
                    note.end = note.start + 0.012

                if include_velocity:  # encode the note confidence in place of velocity
                    velocity = np.median(audio['note'][np.argmin(np.abs(audio['time']-note.start)):
                                                       np.argmin(np.abs(audio['time']-note.end)),
                                         note.pitch-self.labeling.midi_centers[0]])

                    note.velocity = max(1, velocity*127) # velocity should be at least 1 otherwise midi removes the note
                else:
                    velocity = note.velocity/127
                notes.append((note.start, note.end, note.pitch, velocity))
        return notes, midi

    @staticmethod
    def pad_representations(dict_of_representations, pad_length=10):
        """
        Pad the representations so that the DTW does not enforce them to encompass the entire duration.
        Args:
            dict_of_representations: audio or midi representations
            pad_length: how many frames to pad

        Returns:
            padded representations
        """
        for key, value in dict_of_representations.items():
            if key == 'time':
                padded_time = dict_of_representations[key]
                padded_time = np.concatenate([padded_time[:2*pad_length], padded_time+padded_time[2*pad_length]])
                dict_of_representations[key] = padded_time - padded_time[pad_length] # this is to ensure that the
                # first frame times are negative until the real zero time
            elif key in ['onset', 'offset', 'note']:
                dict_of_representations[key] = np.pad(value, ((pad_length, pad_length), (0, 0)))
            elif key in ['start_anchor', 'end_anchor']:
                anchor_time =  dict_of_representations[key][0][0]
                anchor_time = np.argmin(np.abs(dict_of_representations['time'] - anchor_time))
                dict_of_representations[key][:,0] = anchor_time
                dict_of_representations[key] = dict_of_representations[key].astype(np.int)
        return dict_of_representations

    def prepare_for_synchronization(self, audio, midi, feature_rate=44100/256, pad_length=100):
        """
        MrMsDTW works better with start and end anchors. This function finds the start and end anchors for audio
        based on the midi notes. It also pads the MIDI representations since MIDI files most often start with an active
        note and end with an active note. Thus, the DTW will try to align the active notes to the entire duration of the
        audio. This is not desirable. Therefore, we pad the MIDI representations with a few frames of silence at the
        beginning and end of the audio. This way, the DTW will not try to align the active notes to the entire duration.
        Args:
            audio:
            midi:
            feature_rate:
            pad_length:

        Returns:

        """
        # first pad the MIDI
        midi = self.pad_representations(midi, pad_length)

        # sometimes f0s are more reliable than the notes. So, we use both the f0s and the notes together to find the
        # start and end anchors. f0 lookup bins is the number of bins to look around the f0 to assign a note to it.
        f0_lookup_bins = int(100//(2*self.labeling.f0_granularity_c))

        # find the start anchor for the audio
        # first decide on which notes to use for the start anchor (take the entire chord where the MIDI file starts)
        anchor_notes = midi['start_anchor'][:, 1] - self.labeling.midi_centers[0]
        # now find which f0 bins to look at for the start anchor
        anchor_f0s = [self.midi_pitch_to_contour_bin(an+self.labeling.midi_centers[0]) for an in anchor_notes]
        anchor_f0s = np.array([list(range(f0-f0_lookup_bins, f0+f0_lookup_bins+1)) for f0 in anchor_f0s]).reshape(-1)
        # first start anchor proposals come from the notes
        anchor_vals = np.any(audio['note'][:, anchor_notes]>0.5, axis=1)
        # now the f0s
        anchor_vals_f0 = np.any(audio['f0'][:, anchor_f0s]>0.5, axis=1)
        # combine the two
        anchor_vals = np.logical_or(anchor_vals, anchor_vals_f0)
        if not any(anchor_vals):
            return 'corrupted'  # do not consider the file if we cannot find the start anchor
        audio_start = np.argmax(anchor_vals)

        # now the end anchor (most string instruments use chords in cadences: in general the end anchor is polyphonic)
        anchor_notes = midi['end_anchor'][:, 1] - self.labeling.midi_centers[0]
        anchor_f0s = [self.midi_pitch_to_contour_bin(an+self.labeling.midi_centers[0]) for an in anchor_notes]
        anchor_f0s = np.array([list(range(f0-f0_lookup_bins, f0+f0_lookup_bins+1)) for f0 in anchor_f0s]).reshape(-1)
        # the same procedure as above
        anchor_vals = np.any(audio['note'][::-1, anchor_notes]>0.5, axis=1)
        anchor_vals_f0 = np.any(audio['f0'][::-1, anchor_f0s]>0.5, axis=1)
        anchor_vals = np.logical_or(anchor_vals, anchor_vals_f0)
        if not any(anchor_vals):
            return 'corrupted'  # do not consider the file if we cannot find the end anchor
        audio_end = audio['note'].shape[0] - np.argmax(anchor_vals)

        if audio_end - audio_start < (midi['end_anchor'][0][0] - midi['start_anchor'][0][0])/10: # no one plays x10 faster
            return 'corrupted'  # do not consider the interval between anchors is too short
        anchor_pairs = [(audio_start - 5, midi['start_anchor'][0][0] - 5),
                        (audio_end + 5, midi['end_anchor'][0][0] + 5)]

        if anchor_pairs[0][0] < 1:
            anchor_pairs[0] = (1, midi['start_anchor'][0][0])
        if anchor_pairs[1][0] > audio['note'].shape[0] - 1:
            anchor_pairs[1] = (audio['note'].shape[0] - 1, midi['end_anchor'][0][0])

        return audio, midi, [(anchor_pairs[0][0]/feature_rate, anchor_pairs[0][1]/feature_rate),
                             (anchor_pairs[1][0]/feature_rate, anchor_pairs[1][1]/feature_rate)]

