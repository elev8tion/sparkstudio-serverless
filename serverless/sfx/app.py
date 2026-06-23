"""
MusicGen SFX Server
POST /generate — text prompt → ambient audio / sound effect
Uses facebook/musicgen-small — auto-downloads on first request.
"""

import os, io, base64, uuid
from flask import Flask, request, jsonify
import torch
import torchaudio
from transformers import pipeline

app = Flask(__name__)
_pipe = None


def load_model():
    global _pipe
    if _pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        _pipe = pipeline(
            'text-to-audio',
            model='facebook/musicgen-small',
            device=device,
        )
        print(f'[MusicGen] Loaded on {"GPU" if device == 0 else "CPU"}')
    return _pipe


@app.route('/health', methods=['GET'])
def health():
    load_model()
    return jsonify({'status': 'ok', 'model': 'musicgen-sfx'})


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        duration = min(data.get('duration', 5), 30)

        if not prompt:
            return jsonify({'error': 'prompt is required'}), 400

        pipe = load_model()
        seed = data.get('seed', 42)
        generator = torch.Generator(device=pipe.device if hasattr(pipe, 'device') else torch.device('cpu'))
        generator.manual_seed(seed)

        output = pipe(
            prompt,
            forward_params={'do_sample': True, 'max_new_tokens': int(duration * 50)},
        )

        audio_data = output['audio']
        sample_rate = output['sampling_rate']

        waveform = torch.tensor(audio_data).unsqueeze(0)
        out_path = f'/tmp/{uuid.uuid4()}.wav'
        torchaudio.save(out_path, waveform, sample_rate)

        with open(out_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        os.remove(out_path)

        actual_dur = waveform.shape[1] / sample_rate

        return jsonify({
            'audio_base64': audio_b64,
            'mime_type': 'audio/wav',
            'sample_rate': sample_rate,
            'duration_seconds': round(actual_dur, 2),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
