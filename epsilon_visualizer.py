#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import wave
import glob
import os

class EpsilonSpectrogramVisualizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def read_wav_file(self, filename):
        """Read WAV file and return audio data"""
        with wave.open(filename, 'rb') as wav_file:
            # Get audio parameters
            n_channels = wav_file.getnchannels()
            samp_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read audio data
            frames = wav_file.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if samp_width == 1:  # 8-bit
                data = np.frombuffer(frames, dtype=np.uint8) - 128
                data = data.astype(np.float32) / 128.0
            elif samp_width == 2:  # 16-bit
                data = np.frombuffer(frames, dtype=np.int16)
                data = data.astype(np.float32) / 32768.0
            elif samp_width == 3:  # 24-bit
                # Handle 24-bit data
                data = np.frombuffer(frames, dtype=np.uint8)
                data = data.reshape(-1, 3)
                data = ((data[:, 2] << 16) | (data[:, 1] << 8) | data[:, 0])
                data = data.astype(np.float32) / 8388608.0
            else:  # 32-bit or other
                data = np.frombuffer(frames, dtype=np.int32)
                data = data.astype(np.float32) / 2147483648.0
            
            # Handle stereo files by taking only first channel
            if n_channels > 1:
                data = data[::n_channels]
                
            return data, frame_rate
    
    def create_spectrogram(self, audio_data, sample_rate, nfft=1024, overlap=0.75):
        """Create spectrogram using manual STFT calculation"""
        hop_length = int(nfft * (1 - overlap))
        window = np.hanning(nfft)
        
        # Calculate number of frames
        n_frames = 1 + (len(audio_data) - nfft) // hop_length
        
        # Initialize spectrogram matrix
        spectrogram = np.zeros((nfft // 2 + 1, n_frames))
        
        # Compute STFT
        for i in range(n_frames):
            start = i * hop_length
            end = start + nfft
            
            if end > len(audio_data):
                break
                
            # Apply window and compute FFT
            frame = audio_data[start:end] * window
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)
            
            # Store in spectrogram
            spectrogram[:, i] = magnitude
        
        # Convert to dB scale
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        # Time and frequency axes
        time_axis = np.arange(n_frames) * hop_length / sample_rate
        freq_axis = np.fft.rfftfreq(nfft, 1/sample_rate)
        
        return spectrogram_db, time_axis, freq_axis
    
    def plot_spectrogram(self, spectrogram_db, time_axis, freq_axis, title="Spectrogram"):
        """Plot the spectrogram"""
        plt.figure(figsize=(12, 8))
        
        # Plot spectrogram
        extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
        im = plt.imshow(spectrogram_db, aspect='auto', origin='lower', 
                       extent=extent, cmap='viridis')
        
        plt.colorbar(im, label='Intensity (dB)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title(title)
        plt.ylim(0, 5000)  # Limit frequency range for better visibility
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def analyze_epsilon_echo(self, filename):
        """Complete analysis of an epsilon echo WAV file"""
        print(f"Analyzing: {filename}")
        
        # Read audio file
        audio_data, sample_rate = self.read_wav_file(filename)
        duration = len(audio_data) / sample_rate
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Number of samples: {len(audio_data)}")
        
        # Create spectrogram
        spectrogram_db, time_axis, freq_axis = self.create_spectrogram(
            audio_data, sample_rate, nfft=2048, overlap=0.8
        )
        
        # Plot
        title = f"Epsilon Meteor Echo Spectrogram\n{os.path.basename(filename)}"
        plot = self.plot_spectrogram(spectrogram_db, time_axis, freq_axis, title)
        
        # Save plot
        output_filename = os.path.splitext(filename)[0] + '_spectrogram.png'
        plot.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Spectrogram saved as: {output_filename}")
        
        plot.show()
        
        return audio_data, spectrogram_db, time_axis, freq_axis
    
    def batch_process_files(self, file_pattern="epsilon_echo_*.wav"):
        """Process multiple WAV files"""
        wav_files = glob.glob(file_pattern)
        
        if not wav_files:
            print(f"No files found matching pattern: {file_pattern}")
            return
        
        print(f"Found {len(wav_files)} WAV files to process:")
        for file in wav_files:
            print(f"  - {file}")
        
        for file in wav_files:
            try:
                self.analyze_epsilon_echo(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    def create_comparison_plot(self, filenames, output_file="epsilon_comparison.png"):
        """Create comparison plot of multiple spectrograms"""
        n_files = len(filenames)
        fig, axes = plt.subplots(n_files, 1, figsize=(12, 4 * n_files))
        
        if n_files == 1:
            axes = [axes]
        
        for i, filename in enumerate(filenames):
            try:
                audio_data, sample_rate = self.read_wav_file(filename)
                spectrogram_db, time_axis, freq_axis = self.create_spectrogram(
                    audio_data, sample_rate
                )
                
                extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
                im = axes[i].imshow(spectrogram_db, aspect='auto', origin='lower',
                                  extent=extent, cmap='viridis')
                
                axes[i].set_title(f'Epsilon Echo: {os.path.basename(filename)}')
                axes[i].set_ylabel('Frequency (Hz)')
                axes[i].set_ylim(0, 5000)
                axes[i].grid(True, alpha=0.3)
                
                # Add colorbar to last subplot
                if i == n_files - 1:
                    axes[i].set_xlabel('Time (seconds)')
                    cbar = fig.colorbar(im, ax=axes[i], shrink=0.8)
                    cbar.set_label('Intensity (dB)')
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                axes[i].set_title(f"Error: {filename}")
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved as: {output_file}")
        plt.show()

def main():
    # Initialize visualizer
    visualizer = EpsilonSpectrogramVisualizer()
    
    # Process single file
    visualizer.analyze_epsilon_echo('epsilon_meteor_echo.wav')
    
    # Process all epsilon echo files
    visualizer.batch_process_files("epsilon_echo_*.wav")
    
    # Create comparison plot of first few files
    epsilon_files = glob.glob("epsilon_echo_*.wav")[:3]  # First 3 files
    if epsilon_files:
        visualizer.create_comparison_plot(epsilon_files, "epsilon_echoes_comparison.png")

if __name__ == "__main__":
    main()
