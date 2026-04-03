"""
Audio Synthesis Service
Converts MIDI to audio using FluidSynth with configurable soundfonts
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List
import subprocess

try:
    from midi2audio import FluidSynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False
    print("Warning: midi2audio not installed. Run: pip install midi2audio")


# Default soundfont paths (you'll need to download these)
DEFAULT_SOUNDFONTS = {
    "sitar": "soundfonts/sitar.sf2",
    "tanpura": "soundfonts/tanpura.sf2",
    "tabla": "soundfonts/tabla.sf2",
    "piano": "soundfonts/piano.sf2",
    "default": "soundfonts/GeneralUser.sf2"
}


class AudioSynthesizer:
    """
    Synthesizes audio from MIDI files using FluidSynth
    """
    
    def __init__(self, soundfont_dir: str = "soundfonts", sample_rate: int = 44100):
        self.soundfont_dir = Path(soundfont_dir)
        self.sample_rate = sample_rate
        self.available_instruments = self._discover_soundfonts()
        
        if not FLUIDSYNTH_AVAILABLE:
            print("Warning: FluidSynth not available. Audio synthesis disabled.")
    
    def _discover_soundfonts(self) -> List[str]:
        """Discover available soundfont files"""
        instruments = []
        
        if self.soundfont_dir.exists():
            for sf_file in self.soundfont_dir.glob("*.sf2"):
                instrument_name = sf_file.stem.lower()
                instruments.append(instrument_name)
        
        # Add default instruments that should be available
        if not instruments:
            instruments = ["sitar", "piano", "default"]
        
        return instruments
    
    def get_soundfont_path(self, instrument: str) -> Optional[str]:
        """Get soundfont path for instrument"""
        instrument = instrument.lower()
        
        # Check if specific soundfont exists
        sf_path = self.soundfont_dir / f"{instrument}.sf2"
        if sf_path.exists():
            return str(sf_path)
        
        # Try default soundfont
        default_path = self.soundfont_dir / "GeneralUser.sf2"
        if default_path.exists():
            return str(default_path)
        
        # Check if FluidSynth has a default
        return None
    
    def midi_to_audio(
        self,
        midi_path: str,
        output_path: str,
        instrument: str = "sitar",
        volume: float = 1.0
    ) -> Optional[str]:
        """
        Convert MIDI file to audio using FluidSynth
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path for output audio file (WAV)
            instrument: Instrument/soundfont to use
            volume: Volume multiplier (0.0 - 2.0)
            
        Returns:
            Path to output file, or None if failed
        """
        if not FLUIDSYNTH_AVAILABLE:
            print("FluidSynth not available")
            return None
        
        soundfont_path = self.get_soundfont_path(instrument)
        
        if not soundfont_path:
            # Use system FluidSynth with default soundfont
            try:
                fs = FluidSynth()
            except Exception as e:
                print(f"FluidSynth initialization failed: {e}")
                return None
        else:
            try:
                fs = FluidSynth(soundfont_path, sample_rate=self.sample_rate)
            except Exception as e:
                print(f"FluidSynth initialization with {soundfont_path} failed: {e}")
                return None
        
        try:
            fs.midi_to_audio(midi_path, output_path)
            
            # Apply volume adjustment if needed
            if volume != 1.0 and self._has_ffmpeg():
                self._adjust_volume(output_path, volume)
            
            return output_path
            
        except Exception as e:
            print(f"MIDI to audio conversion failed: {e}")
            return None
    
    def _has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _adjust_volume(self, audio_path: str, volume: float):
        """Adjust audio volume using ffmpeg"""
        temp_path = audio_path + ".temp.wav"
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', audio_path,
                '-filter:a', f'volume={volume}',
                temp_path
            ], capture_output=True, check=True)
            
            os.replace(temp_path, audio_path)
        except subprocess.CalledProcessError:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def add_reverb(self, audio_path: str, reverb_amount: float = 0.3) -> Optional[str]:
        """
        Add reverb effect using ffmpeg
        
        Args:
            audio_path: Path to audio file
            reverb_amount: Reverb intensity (0.0 - 1.0)
            
        Returns:
            Path to processed file
        """
        if not self._has_ffmpeg():
            return audio_path
        
        output_path = audio_path.replace('.wav', '_reverb.wav')
        
        try:
            # Simple reverb using aecho filter
            delay = int(reverb_amount * 200)
            decay = reverb_amount * 0.5
            
            subprocess.run([
                'ffmpeg', '-y',
                '-i', audio_path,
                '-filter:a', f'aecho=0.8:0.88:{delay}:{decay}',
                output_path
            ], capture_output=True, check=True)
            
            return output_path
            
        except subprocess.CalledProcessError:
            return audio_path
    
    def normalize_audio(self, audio_path: str) -> Optional[str]:
        """Normalize audio levels using ffmpeg"""
        if not self._has_ffmpeg():
            return audio_path
        
        temp_path = audio_path + ".norm.wav"
        
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', audio_path,
                '-filter:a', 'loudnorm',
                temp_path
            ], capture_output=True, check=True)
            
            os.replace(temp_path, audio_path)
            return audio_path
            
        except subprocess.CalledProcessError:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return audio_path


class MockAudioSynthesizer:
    """
    Mock synthesizer for testing when FluidSynth is not available
    Generates a simple test tone
    """
    
    available_instruments = ["sitar", "piano", "tabla", "tanpura"]
    
    def midi_to_audio(
        self,
        midi_path: str,
        output_path: str,
        instrument: str = "sitar",
        volume: float = 1.0
    ) -> Optional[str]:
        """Generate a simple test tone"""
        try:
            import numpy as np
            import wave
            
            sample_rate = 44100
            duration = 3  # seconds
            frequency = 440  # A4
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t) * volume * 0.5
            
            # Apply envelope
            envelope = np.exp(-t * 0.5)
            tone = tone * envelope
            
            # Convert to 16-bit PCM
            tone = (tone * 32767).astype(np.int16)
            
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(tone.tobytes())
            
            return output_path
            
        except Exception as e:
            print(f"Mock synthesis failed: {e}")
            return None


def get_synthesizer(soundfont_dir: str = "soundfonts") -> AudioSynthesizer:
    """Factory function to get appropriate synthesizer"""
    if FLUIDSYNTH_AVAILABLE:
        return AudioSynthesizer(soundfont_dir)
    else:
        return MockAudioSynthesizer()


if __name__ == "__main__":
    # Test synthesizer
    synth = get_synthesizer()
    print(f"Available instruments: {synth.available_instruments}")
    
    # Test with a sample MIDI file
    test_midi = "all_midi/yaman01_basic_pitch.mid"
    if os.path.exists(test_midi):
        output = synth.midi_to_audio(test_midi, "test_output.wav", "sitar")
        if output:
            print(f"Generated audio: {output}")
        else:
            print("Audio generation failed")
