import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter, filtfilt


class AnimalEventDetector:
    def __init__(self, amplitude_threshold=0.1, min_event_duration=100, sample_rate=44100, band = (1000,2000), ema_alpha=0.2):
        """
        Initialize the detector with parameters.

        Args:
        - amplitude_threshold (float): The amplitude level to consider as an event (0 to 1, normalized).
        - min_event_duration (int): Minimum duration (in milliseconds) for an event.
        - sample_rate (int): The sample rate of the audio (used to normalize time).
        - band (tuple): Frequency range (low_freq, high_freq) to filter out.

        Attributes:
        - audio_array (numpy.ndarray): Array of audio samples.
        - events (list): List of detected events in the audio track.
        """
        self.amplitude_threshold = amplitude_threshold
        self.min_event_duration = min_event_duration  # In milliseconds
        self.sample_rate = sample_rate
        self.band = band
        self.ema_alpha = ema_alpha
        self.ema = []

        self.audio_array = None
        self.events = []

    def load_audio(self, file_path):
        """
        Load an audio file and convert it to a numpy array normalized to the range [-1, 1].

        Args:
        - file_path (str): Path to the audio file.

        Updates:
        - self.audio_array (numpy.ndarray): Normalized audio signal.
        - self.sample_rate (int): Sample rate of the audio file.

        Raises:
        - ValueError: If the file cannot be loaded or processed.
        """
        try:
            # load audio file
            audio = AudioSegment.from_file(file_path)
            print(audio)

            # Get raw samples from the audio file
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            # Normalize the samples to range [-1, 1]
            max_amplitude = np.max(np.abs(samples))
            if max_amplitude > 0:
                samples = samples / max_amplitude

            # Update instance attributes
            self.audio_array = samples
            self.sample_rate = audio.frame_rate  # Store the sample rate for later use

        except Exception as e:
            raise ValueError(f"Problem importing audio file: {e}")

    def detect_events(self, filter=False):
        """
        Detect rising and falling edges in the audio signal based on amplitude threshold.

        Updates:
        - self.events (list): List of tuples [(start_index, end_index), ...] for detected events.

        Raises:
        - ValueError: If audio data is not loaded.
        """

        if self.audio_array is None:
            raise ValueError("Audio data is not loaded. Please load an audio file first.")

        audio = self.audio_array
        if filter:
            audio = self.filter_frequencies(self.audio_array)
        audio_abs = np.abs(audio)  # Take absolute value of the signal
        ema = 0 # Instantiate EMA

        # initialize loop variables
        event_indices = []  # To store (start_index, end_index) pairs
        in_event = False
        start_index = None

        # Loop through signal to detect audio events            
        for i, amplitude in enumerate(audio_abs):
            ema = self.ema_alpha * amplitude + (1-self.ema_alpha) * ema
            self.ema.append(ema)
            if ema > self.amplitude_threshold and not in_event:
                # Rising edge detected
                in_event = True
                start_index = i
            elif ema <= self.amplitude_threshold and in_event:
                # Falling edge detected
                end_index = i
                # find how long the event is in ms
                event_duration_ms = (end_index - start_index) / self.sample_rate * 1000

                # make sure the event is
                if event_duration_ms >= self.min_event_duration:
                    in_event = False
                    event_indices.append((start_index, end_index))

        self.events = event_indices

        return event_indices

    def get_event_timestamps(self):
        """
        Convert event indices to timestamps in seconds.

        Returns:
        - list: List of tuples [(start_time, end_time), ...] in seconds.

        Raises:
        - ValueError: If no events are detected or sample rate is not set.
        """
        if not self.events:
            raise ValueError("No events detected. Please run detect_events() first.")
        if self.sample_rate is None:
            raise ValueError("Sample rate is not set. Please load an audio file first.")

        # Convert indices to timestamps
        return [(start / self.sample_rate, end / self.sample_rate) for start, end in self.events]

    def get_audio_splice(self, start_index, end_index):
        """
        Extract a splice of the audio array based on start and end indices.

        Args:
        - start_index (int): The starting index of the splice.
        - end_index (int): The ending index of the splice.

        Returns:
        - numpy.ndarray: A slice of the audio array between start_index and end_index.

        Raises:
        - ValueError: If indices are invalid or audio data is not loaded.
        """
        if self.audio_array is None:
            raise ValueError("Audio data is not loaded. Please load an audio file first.")
        if not (0 <= start_index <= end_index < len(self.audio_array)):
            raise ValueError(f"Invalid indices: start_index={start_index}, end_index={end_index}")

        return self.audio_array[start_index:end_index]

    def filter_frequencies(self, audio_array):
        """
        Filter out specific frequency components from the audio signal.

        Args:
        - audio_array (numpy.ndarray): The audio signal to filter.

        Returns:
        - filtered_audio: The filtered audio signal.

        Raises:
        - ValueError: If the frequency band is not set or is invalid.
        """
        if not hasattr(self, 'band') or self.band is None:
            raise ValueError("Frequency band for filtering is not set. Please set self.band as (low_freq, high_freq).")
        if self.sample_rate is None:
            raise ValueError("SamplTe rate is not set. Please load an audio file first.")

        low_freq, high_freq = self.band

        # Validate the band frequencies
        if not (0 < low_freq < high_freq < self.sample_rate / 2):
            raise ValueError(
                f"Invalid frequency band: {self.band}. Must satisfy 0 < low_freq < high_freq < Nyquist frequency.")

        nyquist = 0.5 * self.sample_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(N=4, Wn=[low, high], btype='bandstop')

        # Apply the filter to the audio signal
        filtered_audio = filtfilt(b, a, audio_array)

        return filtered_audio

def test():
    print('---testing---')
    # parameters
    file_name = "example.mp3"
    amplitude_threshold = 0.2
    min_event_duration = 200
    sample_rate = 44100
    band = (500, 1500)


    # Initialize detector
    detector = AnimalEventDetector(amplitude_threshold, min_event_duration, sample_rate, band)

    # Load audio
    print('---loading audio data---')
    detector.load_audio(file_name)
    print('---audio successfully loaded---')

    # Detect Events
    print('---detecting events---')
    events = detector.detect_events()
    print(f'---Detected {len(events)} events---')

    # Print Event time stamps
    print('---event timestamps(s)---')
    event_timestamps = detector.get_event_timestamps()
    for i, (start, end) in enumerate(event_timestamps):
        print(f"Event {i + 1}: Start = {start:.2f}s, End = {end:.2f}s")

    # splice audio_array and filter splices
    print('---filtering and splitting audio data---')
    filtered_splices = []

    for i, (start_index, end_index) in enumerate(events):
        # Extract the splice for the event
        splice = detector.get_audio_splice(start_index, end_index)

        # Apply the band-stop filter to the splice
        filtered_splice = detector.filter_frequencies(splice)

        # Append the filtered splice to the list
        filtered_splices.append(filtered_splice)

        print(f"Filtered event {i + 1} processed and added to the list.")
        print(filtered_splice)

    # filtered_splices now contains all the filtered event splices
    print(f"Total filtered splices: {len(filtered_splices)}")

if __name__ == "__main__":
    testing = False
    if testing:
        test()
    