from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple
import pretty_midi
import numpy as np
from musc.postprocessing import RegressionPostProcessor, spotify_create_notes
from musc.pitch_estimator import PitchEstimator


class Transcriber(PitchEstimator):
    def __init__(self, labeling, instrument='Violin', sr=16000, window_size=1024, hop_length=160):
        super().__init__(labeling, instrument=instrument, sr=sr, window_size=window_size, hop_length=hop_length)

    def transcribe(self, audio, batch_size=128, postprocessing='spotify', include_pitch_bends=True, to_midi=True,
                   debug=False):
        """
        Transcribe an audio file or mono waveform in numpy or torch into MIDI with pitch bends.
        :param audio: str, pathlib.Path, np.ndarray, or torch.Tensor
        :param batch_size: frames to process at once
        :param postprocessing: note creation method. 'spotify'(default) or 'tiktok'
        :param include_pitch_bends: whether to include pitch bends in the MIDI file
        :param to_midi: whether to return a MIDI file or a list of note events (as tuple)
        :return: transcribed MIDI file as a pretty_midi.PrettyMIDI object
        """
        out = self.predict(audio, batch_size)
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(out['f0'].T, aspect='auto', origin='lower')
            plt.show()
            plt.imshow(out['note'].T, aspect='auto', origin='lower')
            plt.show()

            plt.imshow(out['onset'].T, aspect='auto', origin='lower')
            plt.show()

            plt.imshow(out['offset'].T, aspect='auto', origin='lower')
            plt.show()

        if to_midi:
            return self.out2midi(out, postprocessing, include_pitch_bends)
        else:
            return self.out2note(out, postprocessing, include_pitch_bends)



    def out2note(self, output: Dict[str, np.array], postprocessing='spotify',
                 include_pitch_bends: bool = True,
    ) -> List[Tuple[float, float, int, float, Optional[List[int]]]]:
        """Convert model output to notes
        """
        if postprocessing == 'spotify':
            estimated_notes = spotify_create_notes(
                output["note"],
                output["onset"],
                note_low=self.labeling.midi_centers[0],
                note_high=self.labeling.midi_centers[-1],
                onset_thresh=0.5,
                frame_thresh=0.3,
                infer_onsets=True,
                min_note_len=int(np.round(127.70 / 1000 * (self.sr / self.hop_length))), #127.70
                melodia_trick=True,
            )
        
        if postprocessing == 'rebab':
            estimated_notes = spotify_create_notes(
                output["note"],
                output["onset"],
                note_low=self.labeling.midi_centers[0],
                note_high=self.labeling.midi_centers[-1],
                onset_thresh=0.2,
                frame_thresh=0.2,
                infer_onsets=True,
                min_note_len=int(np.round(127.70 / 1000 * (self.sr / self.hop_length))), #127.70
                melodia_trick=True,
            )
        
        
        elif postprocessing == 'tiktok':
            postprocessor = RegressionPostProcessor(
                frames_per_second=self.sr / self.hop_length,
                classes_num=self.labeling.midi_centers.shape[0],
                begin_note=self.labeling.midi_centers[0],
                onset_threshold=0.2,
                offset_threshold=0.2,
                frame_threshold=0.3,
                pedal_offset_threshold=0.5,
            )
            tiktok_note_dict, _ = postprocessor.output_dict_to_midi_events(output)
            estimated_notes = []
            for list_item in tiktok_note_dict:
                if list_item['offset_time'] > 0.6 + list_item['onset_time']:
                    estimated_notes.append((int(np.floor(list_item['onset_time']/(output['time'][1]))),
                                            int(np.ceil(list_item['offset_time']/(output['time'][1]))),
                                            list_item['midi_note'], list_item['velocity']/128))
        if include_pitch_bends:
            estimated_notes_with_pitch_bend = self.get_pitch_bends(output["f0"], estimated_notes)
        else:
            estimated_notes_with_pitch_bend = [(note[0], note[1], note[2], note[3], None) for note in estimated_notes]

        times_s = output['time']
        estimated_notes_time_seconds = [
            (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
        ]

        return estimated_notes_time_seconds


    def out2midi(self, output: Dict[str, np.array], postprocessing: str = 'spotify', include_pitch_bends: bool = True,
    ) -> pretty_midi.PrettyMIDI:
        """Convert model output to MIDI
        Args:
            output: A dictionary with shape
                {
                    'frame': array of shape (n_times, n_freqs),
                    'onset': array of shape (n_times, n_freqs),
                    'contour': array of shape (n_times, 3*n_freqs)
                }
                representing the output of the basic pitch model.
            postprocessing: spotify or tiktok postprocessing.
            include_pitch_bends: If True, include pitch bends.
        Returns:
            note_events: A list of note event tuples (start_time_s, end_time_s, pitch_midi, amplitude)
        """
        estimated_notes_time_seconds = self.out2note(output, postprocessing, include_pitch_bends)
        midi_tempo = 120  # todo: infer tempo from the onsets
        return self.note2midi(estimated_notes_time_seconds, midi_tempo)


    def note2midi(
            self, note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
            midi_tempo: float = 120,
    ) -> pretty_midi.PrettyMIDI:
        """Create a pretty_midi object from note events
            :param note_events_with_pitch_bends: list of tuples
                    [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
            :param midi_tempo: #todo: infer tempo from the onsets
            :return: transcribed MIDI file as a pretty_midi.PrettyMIDI object
        """
        mid = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)

        program = pretty_midi.instrument_name_to_program(self.instrument)
        instruments: DefaultDict[int, pretty_midi.Instrument] = defaultdict(
            lambda: pretty_midi.Instrument(program=program)
        )
        for start_time, end_time, note_number, amplitude, pitch_bend in note_events_with_pitch_bends:
            instrument = instruments[note_number]
            note = pretty_midi.Note(
                velocity=int(np.round(127 * amplitude)),
                pitch=note_number,
                start=start_time,
                end=end_time,
            )
            instrument.notes.append(note)
            if not isinstance(pitch_bend, np.ndarray):
                continue
            pitch_bend_times = np.linspace(start_time, end_time, len(pitch_bend))

            for pb_time, pb_midi in zip(pitch_bend_times, pitch_bend):
                instrument.pitch_bends.append(pretty_midi.PitchBend(pb_midi, pb_time))

        mid.instruments.extend(instruments.values())

        return mid

