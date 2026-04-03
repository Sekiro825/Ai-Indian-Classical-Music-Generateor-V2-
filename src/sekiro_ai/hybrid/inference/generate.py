"""
Generation/Inference Module for Hybrid MIDI-Audio Model
Generates expressive Indian Classical Music from text prompts.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np

import torch
import torch.nn.functional as F

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sekiro_ai.hybrid.models.hybrid_cvae import HybridCVAE
from sekiro_ai.hybrid.models.neural_synth import NeuralSynthesizer, SpectrogramSynthesizer
from sekiro_ai.hybrid.config.hybrid_config import HybridCVAEConfig, InferenceConfig
from sekiro_ai.hybrid.musicology import get_raga_grammar, get_taal_name_and_beats

# Import from main project
from sekiro_ai.models.tokenizer import MIDITokenizer


class HybridGenerator:
    """
    High-level interface for generating Indian Classical Music.
    
    Usage:
        generator = HybridGenerator.from_checkpoint("checkpoints/best_model.pt")
        audio, midi = generator.generate(
            raga="yaman",
            mood="peaceful",
            tempo=90,
            duration=60
        )
    """
    
    def __init__(
        self,
        model: HybridCVAE,
        tokenizer: MIDITokenizer,
        synthesizer: Optional[NeuralSynthesizer] = None,
        raga_to_idx: Dict[str, int] = None,
        mood_to_idx: Dict[str, int] = None,
        taal_to_idx: Dict[str, int] = None,
        raga_rules_by_idx: Dict[int, Dict] = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.synthesizer = synthesizer.to(device).eval() if synthesizer else None
        self.raga_to_idx = raga_to_idx or {}
        self.mood_to_idx = mood_to_idx or {}
        self.taal_to_idx = taal_to_idx or {"teental": 0}
        self.raga_rules_by_idx = {int(k): v for k, v in (raga_rules_by_idx or {}).items()}
        self.device = device
        
        # Inverse mappings
        self.idx_to_raga = {v: k for k, v in self.raga_to_idx.items()}
        self.idx_to_mood = {v: k for k, v in self.mood_to_idx.items()}
        self.idx_to_taal = {v: k for k, v in self.taal_to_idx.items()}
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        synth_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: str = 'cuda'
    ) -> 'HybridGenerator':
        """
        Load generator from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            synth_path: Optional path to synthesizer checkpoint
            vocab_path: Optional path to vocabulary mappings
            device: Device to use
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create config and model
        config_dict = checkpoint.get('config', {})
        if isinstance(config_dict, HybridCVAEConfig):
            config = config_dict
        elif isinstance(config_dict, dict):
            config = HybridCVAEConfig(**config_dict) if config_dict else HybridCVAEConfig()
        else:
            config = HybridCVAEConfig()
        
        model = HybridCVAE(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        tokenizer = MIDITokenizer()
        
        # Load synthesizer if available
        synthesizer = None
        if synth_path and Path(synth_path).exists():
            synth_checkpoint = torch.load(synth_path, map_location=device, weights_only=False)
            synthesizer = NeuralSynthesizer()
            synthesizer.load_state_dict(synth_checkpoint['model_state_dict'])
        
        # Load vocabularies
        raga_to_idx = {}
        mood_to_idx = {}
        taal_to_idx = {"teental": 0}
        raga_rules_by_idx = {}
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r') as f:
                vocabs = json.load(f)
                raga_to_idx = vocabs.get('raga_to_idx', {})
                mood_to_idx = vocabs.get('mood_to_idx', {})
                taal_to_idx = vocabs.get('taal_to_idx', {"teental": 0})
                raga_rules_by_idx = vocabs.get('raga_rules_by_idx', {})
        elif raga_to_idx:
            for raga_name, raga_idx in raga_to_idx.items():
                grammar = get_raga_grammar(raga_name, {})
                raga_rules_by_idx[int(raga_idx)] = {
                    "vivadi_pitch_classes": sorted(grammar.vivadi_pitch_classes),
                    "vadi_pitch_classes": sorted(grammar.vadi_pitch_classes),
                    "samvadi_pitch_classes": sorted(grammar.samvadi_pitch_classes),
                    "chalan_degrees": grammar.chalan_degrees,
                }
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            synthesizer=synthesizer,
            raga_to_idx=raga_to_idx,
            mood_to_idx=mood_to_idx,
            taal_to_idx=taal_to_idx,
            raga_rules_by_idx=raga_rules_by_idx,
            device=device
        )
    
    def get_available_ragas(self) -> list:
        """Get list of available ragas"""
        return list(self.raga_to_idx.keys())
    
    def get_available_moods(self) -> list:
        """Get list of available moods"""
        return list(self.mood_to_idx.keys())

    def get_available_taals(self) -> list:
        """Get list of available taals"""
        return list(self.taal_to_idx.keys())

    def _build_raga_token_mask(self, raga_idx: int) -> torch.Tensor:
        """
        Build a token-level mask that blocks vivadi NOTE_ON events for decoding.
        """
        vocab = self.model.config.vocab_size
        mask = torch.ones(vocab, dtype=torch.bool)
        rule = self.raga_rules_by_idx.get(int(raga_idx), {})
        vivadi = rule.get("vivadi_pitch_classes", [])
        if not vivadi:
            return mask

        note_on_offset = self.tokenizer.note_on_offset
        for pc in vivadi:
            for octave in range(11):
                token_id = note_on_offset + int(pc) + 12 * octave
                if 0 <= token_id < vocab:
                    mask[token_id] = False
        # Keep special tokens always valid.
        mask[self.tokenizer.PAD_TOKEN] = True
        mask[self.tokenizer.BOS_TOKEN] = True
        mask[self.tokenizer.EOS_TOKEN] = True
        return mask

    def _build_chalan_prefix(self, raga_idx: int) -> Optional[torch.Tensor]:
        rule = self.raga_rules_by_idx.get(int(raga_idx), {})
        chalan = rule.get("chalan_degrees", [])
        if not chalan:
            return None

        # Build a short NOTE_ON + TIME_SHIFT motif from pitch classes around a central octave.
        tokens: List[int] = [self.tokenizer.BOS_TOKEN]
        base_midi = 60  # Approximate Sa anchor for prompting.
        for degree in chalan[:8]:
            pitch = base_midi + int(degree)
            pitch = max(0, min(127, pitch))
            tokens.append(self.tokenizer.note_on_offset + pitch)
            tokens.append(self.tokenizer.time_shift_offset + 8)
            tokens.append(self.tokenizer.note_off_offset + pitch)
            tokens.append(self.tokenizer.time_shift_offset + 4)
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    @torch.no_grad()
    def generate(
        self,
        raga: str = "yaman",
        mood: str = "peaceful",
        taal: Optional[str] = None,
        tempo: int = 90,
        duration: int = 60,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        enforce_raga_grammar: bool = True,
        use_chalan_prefix: bool = True,
        return_midi: bool = True,
        return_audio: bool = True
    ) -> Dict:
        """
        Generate music based on conditioning.
        
        Args:
            raga: Raga name
            mood: Mood name  
            tempo: Tempo in BPM
            duration: Duration in seconds (approximate)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            return_midi: Whether to return MIDI tokens
            return_audio: Whether to synthesize audio
            
        Returns:
            Dictionary with 'tokens', 'expression', 'midi_path', 'audio' (if synthesized)
        """
        # Convert inputs to indices
        raga_idx = self.raga_to_idx.get(raga.lower(), 0)
        mood_idx = self.mood_to_idx.get(mood.lower(), 0)
        
        if taal is None:
            # Infer a reasonable default taal from raga grammar metadata if available.
            inferred_taal, _ = get_taal_name_and_beats(raga, {})
            taal = inferred_taal
        taal_idx = self.taal_to_idx.get(taal.lower(), self.taal_to_idx.get("teental", 0))

        # Create tensors
        raga_t = torch.tensor([raga_idx], device=self.device)
        mood_t = torch.tensor([mood_idx], device=self.device)
        taal_t = torch.tensor([taal_idx], device=self.device)
        tempo_t = torch.tensor([tempo], device=self.device)
        duration_t = torch.tensor([duration], device=self.device)
        
        # Calculate max tokens based on duration
        # Roughly: 1 second ≈ 20-30 tokens
        max_tokens = min(4096, int(duration * 40))
        min_tokens = int(duration * 25)
        
        token_mask = self._build_raga_token_mask(raga_idx) if enforce_raga_grammar else None
        prefix_tokens = self._build_chalan_prefix(raga_idx) if use_chalan_prefix else None

        # Generate tokens and expression
        tokens, expression = self.model.generate(
            mood=mood_t,
            raga=raga_t,
            taal=taal_t,
            tempo=tempo_t,
            duration=duration_t,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_tokens=prefix_tokens,
            token_mask=token_mask,
            min_length=min_tokens
        )
        
        result = {
            'raga': raga,
            'mood': mood,
            'taal': taal,
            'tempo': tempo,
            'duration': duration
        }
        
        if return_midi:
            # Convert tokens back to MIDI
            token_list = tokens[0].cpu().tolist()
            result['tokens'] = token_list
            
            # Optionally save MIDI file
            # self.tokenizer.detokenize(token_list, output_path)
        
        if expression is not None:
            result['expression'] = expression[0].cpu().numpy()
        
        # Synthesize audio if synthesizer available
        if return_audio and self.synthesizer is not None and expression is not None:
            audio = self.synthesizer(tokens, expression)
            result['audio'] = audio[0].cpu().numpy()
            result['sample_rate'] = 22050
        
        return result
    
    def generate_midi_file(
        self,
        output_path: str,
        raga: str = "yaman",
        mood: str = "peaceful",
        taal: Optional[str] = None,
        tempo: int = 90,
        duration: int = 60,
        **kwargs
    ) -> str:
        """
        Generate and save as MIDI file.
        
        Returns:
            Path to saved MIDI file
        """
        result = self.generate(
            raga=raga,
            mood=mood,
            taal=taal,
            tempo=tempo,
            duration=duration,
            return_midi=True,
            return_audio=False,
            **kwargs
        )
        
        # Detokenize to MIDI
        est_seconds = self.tokenizer.estimate_duration_seconds(result['tokens'])
        if est_seconds > 0:
            time_scale = max(1.0, float(duration) / est_seconds)
        else:
            time_scale = 1.0
        self.tokenizer.detokenize(result['tokens'], output_path, time_scale=time_scale)
        
        return output_path
    
    def generate_audio_file(
        self,
        output_path: str,
        raga: str = "yaman",
        mood: str = "peaceful",
        taal: Optional[str] = None,
        tempo: int = 90,
        duration: int = 60,
        **kwargs
    ) -> Optional[str]:
        """
        Generate and save as audio file.
        
        Returns:
            Path to saved audio file, or None if no synthesizer
        """
        if self.synthesizer is None:
            print("No synthesizer available. Use generate_midi_file instead.")
            return None
        
        result = self.generate(
            raga=raga,
            mood=mood,
            taal=taal,
            tempo=tempo,
            duration=duration,
            return_midi=True,
            return_audio=True,
            **kwargs
        )
        
        if 'audio' in result:
            import soundfile as sf
            sf.write(output_path, result['audio'], result['sample_rate'])
            return output_path
        
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate Indian Classical Music")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--synth", type=str, default=None, help="Synthesizer checkpoint")
    parser.add_argument("--vocab-path", type=str, default=None, help="Path to conditioning_vocabs.json")
    parser.add_argument("--raga", type=str, default="yaman", help="Raga name")
    parser.add_argument("--mood", type=str, default="peaceful", help="Mood")
    parser.add_argument("--taal", type=str, default=None, help="Taal name (optional)")
    parser.add_argument("--tempo", type=int, default=90, help="Tempo BPM")
    parser.add_argument("--duration", type=int, default=60, help="Duration seconds")
    parser.add_argument("--output", type=str, default="generated.mid", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--disable-grammar-mask", action="store_true", help="Disable vivadi masking")
    parser.add_argument("--disable-chalan-prefix", action="store_true", help="Disable chalan prompt prefix")
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("Using CPU")
    
    # Load generator
    print(f"Loading model from {args.checkpoint}...")
    generator = HybridGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        synth_path=args.synth,
        vocab_path=args.vocab_path,
        device=device
    )
    
    # Generate
    print(f"Generating {args.raga} in {args.mood} mood...")
    
    if args.output.endswith('.mid') or args.output.endswith('.midi'):
        output_path = generator.generate_midi_file(
            output_path=args.output,
            raga=args.raga,
            mood=args.mood,
            taal=args.taal,
            tempo=args.tempo,
            duration=args.duration,
            enforce_raga_grammar=not args.disable_grammar_mask,
            use_chalan_prefix=not args.disable_chalan_prefix
        )
        print(f"Saved MIDI: {output_path}")
    else:
        output_path = generator.generate_audio_file(
            output_path=args.output,
            raga=args.raga,
            mood=args.mood,
            taal=args.taal,
            tempo=args.tempo,
            duration=args.duration,
            enforce_raga_grammar=not args.disable_grammar_mask,
            use_chalan_prefix=not args.disable_chalan_prefix
        )
        if output_path:
            print(f"Saved audio: {output_path}")


if __name__ == "__main__":
    main()
