"""
Audio Feature Extraction for Hybrid MIDI-Audio Architecture
Extracts f0 (pitch), amplitude, voiced flags, and spectral centroid from audio.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for audio feature extraction"""
    sample_rate: int = 22050
    hop_length: int = 512  # ~23ms at 22050Hz
    n_fft: int = 2048
    fmin: float = 50.0  # Minimum f0 (Hz) - covers vocal/instrumental range
    fmax: float = 2000.0  # Maximum f0 (Hz)
    chunk_duration: float = 8.0  # seconds per chunk
    

class AudioFeatureExtractor:
    """
    Extract expression features from audio files.
    
    Features extracted:
    - f0: Fundamental frequency (pitch) contour
    - amplitude: RMS energy envelope  
    - voiced: Binary flag indicating voiced frames
    - spectral_centroid: Brightness/timbre indicator
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        
    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load and normalize audio file"""
        y, sr = librosa.load(path, sr=self.config.sample_rate, mono=True)
        y = librosa.util.normalize(y)
        # Trim silence from beginning and end
        y, _ = librosa.effects.trim(y, top_db=30)
        return y, sr
    
    def extract_f0(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency using configured method.
        Returns f0 contour and voiced flag.
        """
        if getattr(self.config, 'use_fast_pitch', False):
            # Extremely fast STFT-based pitch tracking
            pitches, magnitudes = librosa.piptrack(
                y=y,
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )
            # Get index of max magnitude for each frame
            indexes = magnitudes.argmax(axis=0)
            f0 = pitches[indexes, np.arange(magnitudes.shape[1])]
            
            # Voiced if magnitude is reasonably high
            max_mag_per_frame = magnitudes.max(axis=0)
            voiced_flag = max_mag_per_frame > (magnitudes.max() * 0.05)
        else:
            # YIN is much slower but highly accurate
            f0 = librosa.yin(
                y,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                sr=sr,
                hop_length=self.config.hop_length,
                frame_length=self.config.n_fft
            )
            
            # Create voiced flag based on f0 range check
            voiced_flag = (f0 >= self.config.fmin) & (f0 <= self.config.fmax)
        
        # Convert to cents relative to A4 (440 Hz) for better normalization
        f0_cents = np.zeros_like(f0)
        valid_mask = (f0 > 0) & voiced_flag
        
        f0_cents[valid_mask] = 1200 * np.log2(f0[valid_mask] / 440.0 + 1e-8)
        
        # Normalize to roughly [-1, 1] range (typical range is -1200 to +1200 cents)
        f0_normalized = f0_cents / 1200.0
        
        return f0_normalized, voiced_flag.astype(np.float32)

    def estimate_tonic_hz(
        self,
        f0_hz: np.ndarray,
        voiced_flag: np.ndarray
    ) -> float:
        """
        Estimate tonic (Sa) using a robust low-percentile voiced F0 statistic.
        """
        voiced_f0 = f0_hz[(f0_hz > 0) & (voiced_flag > 0)]
        if len(voiced_f0) == 0:
            return 220.0
        tonic = float(np.percentile(voiced_f0, 20))
        return max(self.config.fmin, min(self.config.fmax, tonic))

    def f0_to_sa_relative_cents(
        self,
        f0_hz: np.ndarray,
        voiced_flag: np.ndarray,
        tonic_hz: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert F0 to cents relative to Sa and normalize for model input.
        """
        sa = tonic_hz if tonic_hz and tonic_hz > 0 else self.estimate_tonic_hz(f0_hz, voiced_flag)
        cents = np.zeros_like(f0_hz, dtype=np.float32)
        valid_mask = (f0_hz > 0) & (voiced_flag > 0)
        cents[valid_mask] = 1200.0 * np.log2((f0_hz[valid_mask] + 1e-8) / (sa + 1e-8))
        # Keep most content in [-1, 1] while preserving detailed microtonal movement.
        return (cents / 1200.0).astype(np.float32)
    
    def extract_amplitude(self, y: np.ndarray) -> np.ndarray:
        """Extract RMS energy envelope"""
        rms = librosa.feature.rms(
            y=y, 
            frame_length=self.config.n_fft,
            hop_length=self.config.hop_length
        )[0]
        
        # Normalize to [0, 1]
        rms_normalized = rms / (rms.max() + 1e-8)
        return rms_normalized
    
    def extract_spectral_centroid(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral centroid (brightness)"""
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )[0]
        
        # Normalize by Nyquist frequency
        centroid_normalized = centroid / (sr / 2)
        return centroid_normalized
    
    def extract_all_features(self, path: str, tonic_hz: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Extract all features from an audio file.
        
        Returns:
            Dictionary with keys: 'f0', 'amplitude', 'voiced', 'spectral_centroid'
            Each value is shape (T,) where T is number of frames
        """
        y, sr = self.load_audio(path)
        
        f0_normalized, voiced = self.extract_f0(y, sr)
        amplitude = self.extract_amplitude(y)
        spectral_centroid = self.extract_spectral_centroid(y, sr)

        # Recompute F0 in Hz for Sa-relative cents.
        if getattr(self.config, 'use_fast_pitch', False):
            pitches, magnitudes = librosa.piptrack(
                y=y,
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )
            indexes = magnitudes.argmax(axis=0)
            f0_hz = pitches[indexes, np.arange(magnitudes.shape[1])]
        else:
            f0_hz = librosa.yin(
                y,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                sr=sr,
                hop_length=self.config.hop_length,
                frame_length=self.config.n_fft
            )
        f0_sa_relative = self.f0_to_sa_relative_cents(f0_hz, voiced, tonic_hz=tonic_hz)
        
        # Ensure all features have same length (they should, but just in case)
        min_len = min(len(f0_sa_relative), len(f0_normalized), len(amplitude), len(spectral_centroid))
        
        return {
            'f0': f0_sa_relative[:min_len].astype(np.float32),
            'f0_equal_tempered': f0_normalized[:min_len].astype(np.float32),
            'amplitude': amplitude[:min_len].astype(np.float32),
            'voiced': voiced[:min_len].astype(np.float32),
            'spectral_centroid': spectral_centroid[:min_len].astype(np.float32)
        }
    
    def chunk_features(
        self, 
        features: Dict[str, np.ndarray],
        chunk_frames: int = None
    ) -> list:
        """
        Split features into fixed-size chunks.
        
        Args:
            features: Dictionary of feature arrays
            chunk_frames: Number of frames per chunk (default: 8s worth)
            
        Returns:
            List of feature dictionaries, one per chunk
        """
        if chunk_frames is None:
            # Calculate frames for chunk_duration seconds
            chunk_frames = int(
                self.config.chunk_duration * self.config.sample_rate / self.config.hop_length
            )
        
        total_frames = len(features['f0'])
        chunks = []
        
        for start in range(0, total_frames, chunk_frames):
            end = min(start + chunk_frames, total_frames)
            
            # Skip very short chunks (less than 2 seconds)
            min_frames = int(2.0 * self.config.sample_rate / self.config.hop_length)
            if end - start < min_frames:
                continue
                
            chunk = {
                key: arr[start:end] for key, arr in features.items()
            }
            
            # Pad if necessary
            if end - start < chunk_frames:
                chunk = self._pad_chunk(chunk, chunk_frames)
                
            chunks.append(chunk)
            
        return chunks
    
    def _pad_chunk(
        self, 
        chunk: Dict[str, np.ndarray], 
        target_len: int
    ) -> Dict[str, np.ndarray]:
        """Pad a chunk to target length with zeros"""
        padded = {}
        for key, arr in chunk.items():
            pad_len = target_len - len(arr)
            if pad_len > 0:
                padded[key] = np.pad(arr, (0, pad_len), mode='constant')
            else:
                padded[key] = arr
        return padded
    
    def features_to_tensor(
        self, 
        features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Stack features into a single tensor.
        
        Returns:
            Array of shape (T, 4) with columns [f0, amplitude, voiced, spectral_centroid]
        """
        return np.stack([
            features['f0'],
            features['amplitude'],
            features['voiced'],
            features['spectral_centroid']
        ], axis=-1)


def compute_dataset_statistics(feature_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std for each feature across the dataset.
    Used for z-normalization during training.
    """
    from pathlib import Path
    
    all_features = {
        'f0': [], 'amplitude': [], 
        'voiced': [], 'spectral_centroid': []
    }
    
    feature_path = Path(feature_dir)
    for npy_file in feature_path.glob('*.npy'):
        data = np.load(npy_file, allow_pickle=True).item()
        for key in all_features:
            all_features[key].append(data[key])
    
    stats = {}
    for key, arrays in all_features.items():
        concatenated = np.concatenate(arrays)
        stats[key] = {
            'mean': float(np.mean(concatenated)),
            'std': float(np.std(concatenated) + 1e-8)
        }
    
    return stats


if __name__ == "__main__":
    # Test extraction
    import sys
    
    extractor = AudioFeatureExtractor()
    
    # Test on a sample file
    test_file = Path("d:/MUSIC_MP/EXTRACTED/ALLDATA/DData/yaman01.wav")
    
    if test_file.exists():
        print(f"Extracting features from: {test_file}")
        features = extractor.extract_all_features(str(test_file))
        
        print(f"\nFeature shapes:")
        for key, arr in features.items():
            print(f"  {key}: {arr.shape}")
        
        # Test chunking
        chunks = extractor.chunk_features(features)
        print(f"\nGenerated {len(chunks)} chunks")
        
        # Test tensor conversion
        tensor = extractor.features_to_tensor(chunks[0])
        print(f"Tensor shape: {tensor.shape}")
    else:
        print(f"Test file not found: {test_file}")
