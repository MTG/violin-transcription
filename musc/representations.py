from mir_eval import melody
import numpy as np
from scipy.stats import norm
import librosa
import pretty_midi
from scipy.ndimage import gaussian_filter1d


class PerformanceLabel:
    """
    The dataset labeling class for performance representations. Currently, includes onset, note, and fine-grained f0
    representations. Note min, note max, and f0_bin_per_semitone values are to be arranged per instrument. The default
    values are for violin performance analysis. Fretted instruments might not require such f0 resolutions per semitone.
    """
    def __init__(self, note_min='F#3', note_max='C8', f0_bins_per_semitone=9, f0_smooth_std_c=None,
                 onset_smooth_std=0.7, f0_tolerance_c=200):
        midi_min = pretty_midi.note_name_to_number(note_min)
        midi_max = pretty_midi.note_name_to_number(note_max)
        self.midi_centers = np.arange(midi_min, midi_max)
        self.onset_smooth_std=onset_smooth_std # onset smoothing along time axis (compensate for alignment)

        f0_hz_range = librosa.note_to_hz([note_min, note_max])
        f0_c_min, f0_c_max = melody.hz2cents(f0_hz_range)
        self.f0_granularity_c = 100/f0_bins_per_semitone
        if not f0_smooth_std_c:
            f0_smooth_std_c = self.f0_granularity_c * 5/4  # Keep the ratio from the CREPE paper (20 cents and 25 cents)
        self.f0_smooth_std_c = f0_smooth_std_c

        self.f0_centers_c = np.arange(f0_c_min, f0_c_max, self.f0_granularity_c)
        self.f0_centers_hz = 10 * 2 ** (self.f0_centers_c / 1200)
        self.f0_n_bins = len(self.f0_centers_c)

        self.pdf_normalizer = norm.pdf(0)

        self.f0_c2hz = lambda c: 10*2**(c/1200)
        self.f0_hz2c = melody.hz2cents
        self.midi_centers_c = self.f0_hz2c(librosa.midi_to_hz(self.midi_centers))

        self.f0_tolerance_bins = int(f0_tolerance_c/self.f0_granularity_c)
        self.f0_transition_matrix = gaussian_filter1d(np.eye(2*self.f0_tolerance_bins + 1), 25/self.f0_granularity_c)

    def f0_c2label(self, pitch_c):
        """
        Convert a single f0 value in cents to a one-hot label vector with smoothing (i.e., create a gaussian blur around
        the target f0 bin for regularization and training stability. The blur is controlled by self.f0_smooth_std_c
        :param pitch_c: a single pitch value in cents
        :return: one-hot label vector with frequency blur
        """
        result = norm.pdf((self.f0_centers_c - pitch_c) / self.f0_smooth_std_c).astype(np.float32)
        result /= self.pdf_normalizer
        return result

    def f0_label2c(self, salience, center=None):
        """
        Convert the salience predictions to monophonic f0 in cents. Only outputs a single f0 value per frame!
        :param salience: f0 activations
        :param center: f0 center bin to calculate the weighted average. Use argmax if empty
        :return: f0 array per frame (in cents).
        """
        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(salience * self.f0_centers_c[start:end])
            weight_sum = np.sum(salience)
            return product_sum / np.clip(weight_sum, 1e-8, None)
        if salience.ndim == 2:
            return np.array([self.f0_label2c(salience[i, :]) for i in range(salience.shape[0])])
        raise Exception("label should be either 1d or 2d ndarray")

    def fill_onset_matrix(self, onsets, window, feature_rate):
        """
        Create a sparse onset matrix from window and onsets (per-semitone). Apply a gaussian smoothing (along time)
        so that we can tolerate better the alignment problems. This is similar to the frequency smoothing for the f0.
        The temporal smoothing is controlled by the parameter self.onset_smooth_std
        :param onsets: A 2d np.array of individual note onsets with their respective time values
        (Nx2: time in seconds - midi number)
        :param window: Timestamps for the frame centers of the sparse matrix
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds
        :return: onset_roll: A sparse matrix filled with temporally blurred onsets.
        """
        onsets = self.get_window_feats(onsets, window, feature_rate)
        onset_roll = np.zeros((len(window), len(self.midi_centers)))
        for onset in onsets:
            onset, note = onset  # it was a pair with time and midi note
            if self.midi_centers[0] < note < self.midi_centers[-1]: # midi note should be in the range defined
                note = int(note) - self.midi_centers[0]  # find the note index in our range
                onset = (onset*feature_rate)-window[0]    # onset index (as float but in frames, not in seconds!)
                start = max(0, int(onset) - 3)
                end = min(len(window) - 1, int(onset) + 3)
                try:
                    vals = norm.pdf(np.linspace(start - onset, end - onset, end - start + 1) / self.onset_smooth_std)
                    # if you increase 0.7 you smooth the peak
                    # if you decrease it, e.g., 0.1, it becomes too peaky! around 0.5-0.7 seems ok
                    vals /= self.pdf_normalizer
                    onset_roll[start:end + 1, note] += vals
                except ValueError:
                    print('start',start, 'onset', onset, 'end', end)
        return onset_roll, onsets

    def fill_note_matrix(self, notes, window, feature_rate):
        """
        Create the note matrix (piano roll) from window timestamps and note values per frame.
        :param notes: A 2d np.array of individual notes with their active time values Nx2
        :param window: Timestamps for the frame centers of the output
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds
        :return note_roll: The piano roll in the defined range of [note_min, note_max).
        """
        notes = self.get_window_feats(notes, window, feature_rate)

        # take the notes in the midi range defined
        notes = notes[np.logical_and(notes[:,1]>=self.midi_centers[0], notes[:,1]<=self.midi_centers[-1]),:]

        times = (notes[:,0]*feature_rate - window[0]).astype(int) # in feature samples (fs:self.hop/self.sr)
        notes = (notes[:,1] - self.midi_centers[0]).astype(int)

        note_roll = np.zeros((len(window), len(self.midi_centers)))
        note_roll[(times, notes)] = 1
        return note_roll, notes


    def fill_f0_matrix(self, f0s, window, feature_rate):
        """
        Unlike the labels for onsets and notes, f0 label is only relevant for strictly monophonic regions! Thus, this
        function returns a boolean which represents where to apply the given values.
        Never back-propagate without the boolean! Empty frames mean that the label is not that reliable.

        :param f0s: A 2d np.array of f0 values with the time they belong to (2xN: time in seconds - f0 in Hz)
        :param window: Timestamps for the frame centers of the output
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds

        :return f0_roll: f0 label matrix and
                f0_hz: f0 values in Hz
                annotation_bool: A boolean array representing which frames have reliable f0 annotations.
        """
        f0s = self.get_window_feats(f0s, window, feature_rate)
        f0_cents = np.zeros_like(window, dtype=float)
        f0s[:,1] = self.f0_hz2c(f0s[:,1]) # convert f0 in hz to cents

        annotation_bool = np.zeros_like(window, dtype=bool)
        f0_roll = np.zeros((len(window), len(self.f0_centers_c)))
        times_in_frame = f0s[:, 0]*feature_rate - window[0]
        for t, f0 in enumerate(f0s):
            t = times_in_frame[t]
            if t%1 < 0.25: # only consider it as annotation if the f0 values is really close to the frame center
                t = int(np.round(t))
                f0_roll[t] = self.f0_c2label(f0[1])
                annotation_bool[t] = True
                f0_cents[t] = f0[1]

        return f0_roll, f0_cents, annotation_bool


    @staticmethod
    def get_window_feats(time_feature_matrix, window, feature_rate):
        """
        Restrict the feature matrix to the features that are inside the window
        :param window: Timestamps for the frame centers of the output
        :param time_feature_matrix: A 2d array of Nx2 per the entire file.
        :param feature_rate: Window timestamps are integer, this is to convert them to seconds
        :return: window_features: the features inside the given window
        """
        start = time_feature_matrix[:,0]>(window[0]-0.5)/feature_rate
        end = time_feature_matrix[:,0]<(window[-1]+0.5)/feature_rate
        window_features = np.logical_and(start, end)
        window_features = np.array(time_feature_matrix[window_features,:])
        return window_features

    def represent_midi(self, midi, feature_rate):
        """
        Represent a midi file as sparse matrices of onsets, offsets, and notes. No f0 is included.
        :param midi: A midi file (either a path or a pretty_midi.PrettyMIDI object)
        :param feature_rate: The feature rate in Hz
        :return: dict {onset, offset, note, time}: Same format with the model's learning and outputs
        """
        def _get_onsets_offsets_frames(midi_content):
            if isinstance(midi_content, str):
                midi_content = pretty_midi.PrettyMIDI(midi_content)
            onsets = []
            offsets = []
            frames = []
            for instrument in midi_content.instruments:
                for note in instrument.notes:
                    start = int(np.round(note.start * feature_rate))
                    end = int(np.round(note.end * feature_rate))
                    note_times = (np.arange(start, end+0.5)/feature_rate)[:, np.newaxis]
                    note_pitch = np.full_like(note_times, fill_value=note.pitch)
                    onsets.append([note.start, note.pitch])
                    offsets.append([note.end, note.pitch])
                    frames.append(np.hstack([note_times, note_pitch]))
            onsets = np.vstack(onsets)
            offsets = np.vstack(offsets)
            frames = np.vstack(frames)
            return onsets, offsets, frames, midi_content
        onset_array, offset_array, frame_array, midi_object = _get_onsets_offsets_frames(midi)
        window = np.arange(frame_array[0, 0]*feature_rate, frame_array[-1, 0]*feature_rate, dtype=int)
        onset_roll, _ = self.fill_onset_matrix(onset_array, window, feature_rate)
        offset_roll, _ = self.fill_onset_matrix(offset_array, window, feature_rate)
        note_roll, _ = self.fill_note_matrix(frame_array, window, feature_rate)
        start_anchor = onset_array[onset_array[:, 0]==np.min(onset_array[:, 0])]
        end_anchor = offset_array[offset_array[:, 0]==np.max(offset_array[:, 0])]
        return {
            'midi': midi_object,
            'note': note_roll,
            'onset': onset_roll,
            'offset': offset_roll,
            'time': window/feature_rate,
            'start_anchor': start_anchor,
            'end_anchor': end_anchor
        }
