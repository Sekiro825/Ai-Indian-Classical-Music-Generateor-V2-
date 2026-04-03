"""
LLM Integration for parsing user text input
Supports OpenRouter API (with fallback to mock parser)
Extracts mood, tempo, raga, and duration from natural language descriptions or lyrics
"""

import os
import json
import re
from typing import Dict, Optional, List
from dataclasses import dataclass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not installed. Run: pip install requests")


@dataclass
class MusicParameters:
    """Extracted music generation parameters"""
    mood: str
    tempo: int  # BPM
    duration: int  # seconds
    raga: str
    confidence: float = 0.0
    raw_response: str = ""


# Raga-mood mapping for LLM context
RAGA_MOOD_MAPPING = {
    "romantic": ["yaman", "bageshree", "des"],
    "peaceful": ["yaman", "bhoopali", "bhupali"],
    "sad": ["bhairavi", "asavari", "darbari"],
    "melancholic": ["bhairavi", "asavari"],
    "devotional": ["bhairavi", "bhoopali", "bhajani"],
    "mysterious": ["malkauns", "darbari"],
    "meditative": ["malkauns", "bhairavi"],
    "energetic": ["sarang", "jhaptal"],
    "joyful": ["sarang", "bhoopali"],
    "serious": ["malkauns", "darbari", "asavari"],
    "calm": ["yaman", "bhoopali"],
    "festive": ["sarang", "trital"],
    "royal": ["darbari"],
    "playful": ["dadra", "des"]
}

# Tempo guidelines
TEMPO_MAPPING = {
    "very slow": 45,
    "slow": 60,
    "moderate": 90,
    "medium": 100,
    "fast": 130,
    "very fast": 160,
    "vilambit": 45,  # Hindustani terms
    "madhya": 90,
    "drut": 140
}


class OpenRouterParser:
    """
    Uses OpenRouter API to parse natural language into music generation parameters
    """
    
    SYSTEM_PROMPT = """You are an expert in Indian classical music (Hindustani tradition).
Given a user's description or lyrics, extract the following parameters for music generation:

1. **mood**: The emotional quality (choose from: romantic, peaceful, sad, melancholic, devotional, mysterious, meditative, energetic, joyful, serious, calm, festive, royal, playful)

2. **tempo**: Speed in BPM (40-180). Consider:
   - Vilambit (slow): 40-60 BPM
   - Madhya (medium): 60-100 BPM  
   - Drut (fast): 100-180 BPM

3. **duration**: Length in seconds (30-300). Consider the complexity of the description.

4. **raga**: The most suitable raga:
   - Yaman: romantic, peaceful evening
   - Bhairavi: devotional, melancholic morning
   - Malkauns: mysterious, meditative night
   - Bageshree: romantic longing
   - Bhoopali: peaceful, happy evening
   - Asavari: sad, contemplative
   - Darbari: royal, serious night
   - Sarang: energetic, joyful noon

Respond ONLY in this exact JSON format:
{"mood": "...", "tempo": 90, "duration": 60, "raga": "...", "reasoning": "..."}
"""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: Optional[str] = None):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package not installed")
        
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not provided")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Required by OpenRouter
            "X-Title": "Raga Music Generator"
        }
        # Use a fast, cheap model
        self.model = "google/gemini-2.0-flash-001"
    
    def parse_text(self, user_input: str) -> MusicParameters:
        """
        Parse user text input into music generation parameters
        
        Args:
            user_input: User's description or lyrics
            
        Returns:
            MusicParameters with extracted values
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f'Extract music parameters from: "{user_input}"'}
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }
        
        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            raw_text = result['choices'][0]['message']['content'].strip()
            
            # Extract JSON from response
            params = self._extract_json(raw_text)
            
            return MusicParameters(
                mood=params.get('mood', 'calm'),
                tempo=int(params.get('tempo', 90)),
                duration=int(params.get('duration', 60)),
                raga=params.get('raga', 'yaman'),
                confidence=0.9,
                raw_response=raw_text
            )
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return self._fallback_parse(user_input)
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: try whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    
    def _fallback_parse(self, user_input: str) -> MusicParameters:
        """
        Simple keyword-based fallback when API fails
        """
        text_lower = user_input.lower()
        
        # Detect mood
        mood = "calm"
        for m, keywords in [
            ("sad", ["sad", "cry", "tears", "sorrow", "pain", "broken"]),
            ("romantic", ["love", "romance", "heart", "beloved", "passion"]),
            ("peaceful", ["peace", "calm", "serene", "quiet", "gentle"]),
            ("energetic", ["dance", "fast", "energy", "power", "strength"]),
            ("devotional", ["god", "divine", "prayer", "worship", "spiritual"]),
            ("joyful", ["happy", "joy", "celebrate", "cheerful"])
        ]:
            if any(kw in text_lower for kw in keywords):
                mood = m
                break
        
        # Detect tempo from keywords
        tempo = 90
        for t_word, t_val in TEMPO_MAPPING.items():
            if t_word in text_lower:
                tempo = t_val
                break
        
        # Get raga from mood
        raga_options = RAGA_MOOD_MAPPING.get(mood, ["yaman"])
        raga = raga_options[0]
        
        # Estimate duration
        word_count = len(user_input.split())
        duration = min(max(30, word_count * 3), 180)
        
        return MusicParameters(
            mood=mood,
            tempo=tempo,
            duration=duration,
            raga=raga,
            confidence=0.5,
            raw_response="fallback_parser"
        )


class MockGeminiParser:
    """
    Mock parser for testing without API key
    """
    
    def parse_text(self, user_input: str) -> MusicParameters:
        """Mock parsing - uses keyword matching"""
        text_lower = user_input.lower()
        
        # Simple mood detection
        if any(w in text_lower for w in ["sad", "cry", "tears", "sorrow"]):
            return MusicParameters("sad", 60, 90, "bhairavi", 0.8, "mock")
        elif any(w in text_lower for w in ["love", "romance", "heart"]):
            return MusicParameters("romantic", 80, 120, "yaman", 0.8, "mock")
        elif any(w in text_lower for w in ["peace", "calm", "meditation"]):
            return MusicParameters("peaceful", 50, 90, "malkauns", 0.8, "mock")
        elif any(w in text_lower for w in ["dance", "fast", "energy"]):
            return MusicParameters("energetic", 140, 60, "sarang", 0.8, "mock")
        elif any(w in text_lower for w in ["god", "divine", "prayer"]):
            return MusicParameters("devotional", 70, 120, "bhairavi", 0.8, "mock")
        else:
            return MusicParameters("calm", 90, 60, "yaman", 0.5, "mock")


# Backwards compatibility alias
GeminiParser = OpenRouterParser


def get_parser(api_key: Optional[str] = None) -> 'OpenRouterParser':
    """
    Factory function to get appropriate parser
    Checks for OPENROUTER_API_KEY first, then falls back to provided key
    """
    # Check for OpenRouter key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY") or api_key
    
    if openrouter_key and openrouter_key.startswith("sk-or-"):
        try:
            return OpenRouterParser(openrouter_key)
        except Exception as e:
            print(f"OpenRouter init failed: {e}")
    
    # Fallback to mock
    print("Using mock parser (no valid API key)")
    return MockGeminiParser()


if __name__ == "__main__":
    # Test the parser
    parser = get_parser()
    
    test_inputs = [
        "I want a sad song about heartbreak and loss",
        "Create something peaceful for meditation",
        "A romantic melody for a wedding evening",
        "Fast energetic music for dancing",
        "मुझे एक भक्ति गीत चाहिए (I want a devotional song)"
    ]
    
    for input_text in test_inputs:
        result = parser.parse_text(input_text)
        print(f"\nInput: {input_text}")
        print(f"Result: mood={result.mood}, tempo={result.tempo}, duration={result.duration}s, raga={result.raga}")
