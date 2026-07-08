"""
Speaker diarization using ivrit.ai's fork of pyannote's standard
speaker-diarization-3.1 pipeline (in-call diarization: who spoke when within
THIS audio file). This is deliberately separate from speaker_processing.py,
which does cross-call voice-profile matching via raw embeddings and is not
used by rp_handler.py.
"""

import os
import threading

from pyannote.audio import Pipeline

_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    """Lazily loads and caches the diarization pipeline (once per warm worker)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                _pipeline = Pipeline.from_pretrained(
                    "ivrit-ai/pyannote-speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
    return _pipeline


def diarize(audio_path):
    """
    Runs speaker diarization on the given audio file.

    Returns a list of {"start": float, "end": float, "speaker": str} turns,
    sorted by start time.
    """
    pipeline = _get_pipeline()
    annotation = pipeline(audio_path)
    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    return turns


def assign_speakers(segments, turns):
    """
    Assigns a speaker label to each Whisper segment (mutated in place) based
    on the diarization turn with the greatest time overlap. A segment with no
    overlapping turn is left without a "speaker" key.
    """
    for segment in segments:
        seg_start, seg_end = segment["start"], segment["end"]
        best_turn = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
        if best_turn is not None:
            segment["speaker"] = best_turn["speaker"]
    return segments
