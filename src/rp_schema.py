INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': False,
        'default': None
    },
    'audio_base64': {
        'type': str,
        'required': False,
        'default': None
    },
    'download_headers': {
        'type': dict,
        'required': False,
        'default': None
    },
    'model': {
        'type': str,
        'required': False,
        'default': 'small'
    },
    'task': {
        'type': str,
        'required': False,
        'default': 'transcribe'
    },
    'device': {
        'type': str,
        'required': False,
        'default': "cuda"
    },
    'device_index': {
        'type': int,
        'required': False,
        'default': 0
    },
    'language': {
        'type': str,
        'required': False,
        'default': None
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 8
    },
    'compute_type': {
        'type': str,
        'required': False,
        'default': "float32"
    },
    'best_of': {
        'type': int,
        'required': False,
        'default': 5
    },
    'beam_size': {
        'type': int,
        'required': False,
        'default': 5
    },
    'threads': {
        'type': int,
        'required': False,
        'default': 0
    },
    'chunk_size': {
        'type': int,
        'required': False,
        'default': 30
    },
    'patience': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'length_penalty': {
        'type': float,
        'required': False,
        'default': 0
    },
    'suppress_tokens': {
        'type': str,
        'required': False,
        'default': '-1'
    },
    'initial_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'condition_on_previous_text': {
        'type': bool,
        'required': False,
        'default': True
    },
    'temperature_increment_on_fallback': {
        'type': float,
        'required': False,
        'default': 0.2
    },
    'compression_ratio_threshold': {
        'type': float,
        'required': False,
        'default': 2.4
    },
    'logprob_threshold': {
        'type': float,
        'required': False,
        'default': -1.0
    },
    'no_speech_threshold': {
        'type': float,
        'required': False,
        'default': 0.6
    },
    'enable_vad': {
        'type': bool,
        'required': False,
        'default': True
    },
    'vad_method': {
        'type': str,
        'required': False,
        'default': "pyannote"
    },
    'vad_onset': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    'vad_offset': {
        'type': float,
        'required': False,
        'default': 0.363
    },
    'align_model': {
        'type': str,
        'required': False,
        'default': None
    },
    'word_timestamps': {
        'type': bool,
        'required': False,
        'default': True
    },
    'return_char_alignments': {
        'type': bool,
        'required': False,
        'default': False
    },
    'no_align': {
        'type': bool,
        'required': False,
        'default': False
    },
    'diarize': {
        'type': bool,
        'required': False,
        'default': False
    },
    'diarize_model': {
        'type': str,
        'required': False,
        'default': "pyannote"
    },
}