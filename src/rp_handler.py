"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import tempfile

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger, rp_download
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict
import diarization


MODEL = predict.Predictor()
MODEL.setup()


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    global audio_input
    job_input = job['input']

    audio_base64 = job_input.pop("audio_base64", False)
    print(job_input)
    input_validation = validate(job_input, INPUT_VALIDATIONS)

    if 'errors' in input_validation:
        return {"error": input_validation['errors']}
    job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not audio_base64:
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and audio_base64:
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        rp_download.HEADERS = job_input.get("download_headers", rp_download.HEADERS)
        audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if audio_base64:
        audio_input = base64_to_tempfile(audio_base64)

    whisper_results = MODEL.predict(
        audio=audio_input,
        model_name=job_input["model"],
        compute_type=job_input["compute_type"],
        language=job_input["language"],
        temperature=job_input["temperature"],
        best_of=job_input["best_of"],
        beam_size=job_input["beam_size"],
        patience=job_input["patience"],
        length_penalty=job_input["length_penalty"],
        suppress_tokens=job_input.get("suppress_tokens", "-1"),
        initial_prompt=job_input["initial_prompt"],
        condition_on_previous_text=job_input["condition_on_previous_text"],
        temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
        compression_ratio_threshold=job_input["compression_ratio_threshold"],
        logprob_threshold=job_input["logprob_threshold"],
        no_speech_threshold=job_input["no_speech_threshold"],
        enable_vad=job_input["enable_vad"],
        word_timestamps=job_input["word_timestamps"]
    )

    if job_input.get("diarize"):
        turns = diarization.diarize(audio_input)
        whisper_results["segments"] = diarization.assign_speakers(
            whisper_results["segments"], turns
        )

    rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})