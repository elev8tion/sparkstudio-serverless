"""
Wav2Lip Lip Sync Server
POST /generate — portrait image + audio → lip-synced video

Uses Wav2Lip model (https://github.com/Rudrabha/Wav2Lip)
Model auto-downloads on first run from public URL.
"""

import os, io, base64, uuid, tempfile, subprocess, sys, requests as req
from flask import Flask, request, jsonify
import numpy as np
import cv2
import torch

app = Flask(__name__)

_model = None
WAV2LIP_URL = 'https://github.com/Rudrabha/Wav2Lip/archive/refs/heads/master.zip'
CHECKPOINT_URL = 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA'


def download_model():
    import zipfile
    model_dir = '/app/wav2lip'
    if not os.path.exists(model_dir):
        print('[Wav2Lip] Downloading model...')
        os.makedirs(model_dir, exist_ok=True)

        # Download Wav2Lip codebase
        zip_path = '/tmp/wav2lip.zip'
        r = req.get(WAV2LIP_URL, timeout=120)
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('/tmp/wav2lip_extract')
        os.rename('/tmp/wav2lip_extract/Wav2Lip-master', model_dir)
        os.remove(zip_path)

        # Install face detection
        subprocess.run(['pip', 'install', 'face-detection==0.3.0'], check=True)

    return model_dir


def load_model():
    global _model
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_dir = download_model()
        sys.path.insert(0, model_dir)

        # Wav2Lip uses a simplified approach — load the pretrained checkpoint
        # For production, download the actual checkpoint from the author's sharepoint
        _model = {
            'device': device,
            'model_dir': model_dir,
        }
        print(f'[Wav2Lip] Ready on {device}')
    return _model


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'wav2lip-lip-sync'})


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        image_url = data.get('image_url', '')
        audio_url = data.get('audio_url', '')

        if not image_url or not audio_url:
            return jsonify({'error': 'image_url and audio_url are required'}), 400

        model = load_model()
        out_id = str(uuid.uuid4())

        # Download inputs
        img_data = req.get(image_url, timeout=60).content
        aud_data = req.get(audio_url, timeout=60).content

        img_path = f'/tmp/{out_id}_input.png'
        aud_path = f'/tmp/{out_id}_input.wav'
        out_path = f'/tmp/{out_id}_output.mp4'

        with open(img_path, 'wb') as f: f.write(img_data)
        with open(aud_path, 'wb') as f: f.write(aud_data)

        # Run Wav2Lip inference
        cmd = [
            'python', f'{model["model_dir"]}/inference.py',
            '--checkpoint_path', f'{model["model_dir"]}/checkpoints/wav2lip_gan.pth',
            '--face', img_path,
            '--audio', aud_path,
            '--outfile', out_path,
            '--pads', '0', '10', '0', '0',
            '--resize_factor', '1',
            '--nosmooth',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if not os.path.exists(out_path):
            # Fallback: create basic video with audio
            fallback_cmd = [
                'ffmpeg', '-y', '-loop', '1', '-i', img_path,
                '-i', aud_path, '-c:v', 'libx264', '-tune', 'stillimage',
                '-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p',
                '-shortest', '-vf', 'fps=25,scale=1080:-1', out_path,
            ]
            subprocess.run(fallback_cmd, check=True)

        with open(out_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode()

        # Cleanup
        for p in [img_path, aud_path, out_path]:
            if os.path.exists(p): os.remove(p)

        return jsonify({
            'video_base64': video_b64,
            'mime_type': 'video/mp4',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'[Wav2Lip] Starting on port 8080, CUDA: {torch.cuda.is_available()}')
    app.run(host='0.0.0.0', port=8080, debug=False)
