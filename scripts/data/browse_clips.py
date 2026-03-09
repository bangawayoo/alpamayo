#!/usr/bin/env python3
"""Lightweight web clip browser using Flask.

All interactions use fetch() — no page navigation, so it works behind
Kubeflow/JupyterHub reverse proxies without losing the connection.

Usage:
    python scripts/data/browse_clips.py
    python scripts/data/browse_clips.py --port 7860
"""
import argparse
import base64
import io
import json
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

_avdi = None
_valid_clips = None


def _init_dataset():
    global _avdi, _valid_clips
    if _avdi is not None:
        return
    from physical_ai_av import PhysicalAIAVDatasetInterface

    _avdi = PhysicalAIAVDatasetInterface()
    clip_index = _avdi.clip_index
    _valid_clips = clip_index[
        (clip_index["split"] == "train") & clip_index["clip_is_valid"]
    ]


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def load_clip_frames(clip_id: str, t0_us: int):
    _init_dataset()
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    data = load_physical_aiavdataset(
        clip_id=clip_id, t0_us=t0_us, avdi=_avdi, maybe_stream=True
    )
    frames = data["image_frames"]
    n_cameras, n_frames = frames.shape[0], frames.shape[1]

    images = []
    for c in range(n_cameras):
        for f in range(n_frames):
            img = Image.fromarray(
                frames[c, f].permute(1, 2, 0).numpy().astype(np.uint8)
            )
            img.thumbnail((480, 270), Image.LANCZOS)
            images.append({
                "label": f"cam{c}_frame{f}",
                "b64": _img_to_b64(img),
            })
    return images, n_cameras, n_frames


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Alpamayo Clip Browser</title>
<style>
  body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a1a; color: #eee; }
  h1 { margin-bottom: 5px; }
  .controls { display: flex; gap: 10px; margin: 15px 0; align-items: end; }
  .controls input[type=text] { flex: 3; padding: 8px; font-size: 14px; font-family: monospace; background: #2a2a2a; color: #eee; border: 1px solid #555; }
  .controls input[type=number] { flex: 1; padding: 8px; font-size: 14px; background: #2a2a2a; color: #eee; border: 1px solid #555; }
  .controls button { padding: 8px 16px; font-size: 14px; cursor: pointer; background: #3b82f6; color: white; border: none; border-radius: 4px; }
  .controls button:hover { background: #2563eb; }
  .info { background: #2a2a2a; padding: 10px; margin: 10px 0; font-family: monospace; font-size: 13px; border-radius: 4px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 15px 0; }
  .grid .cell { text-align: center; }
  .grid img { width: 100%; border-radius: 4px; }
  .grid .label { font-size: 11px; color: #aaa; margin-top: 3px; }
  .clip-list { background: #2a2a2a; padding: 10px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; border-radius: 4px; display: none; }
  .clip-list span { cursor: pointer; display: block; padding: 2px 0; }
  .clip-list span:hover { color: #3b82f6; }
  .error { color: #f87171; margin: 10px 0; }
  .loading { color: #aaa; margin: 10px 0; }
</style>
</head>
<body>
<h1>Alpamayo Clip Browser</h1>
<p style="color:#aaa;margin-top:0">Enter a clip_id to view camera frames.</p>

<div class="controls">
  <input type="text" id="clip_id" placeholder="e.g. 25cd4769-5dcf-4b53-a351-bf2c5deb6124">
  <input type="number" id="t0_us" value="5100000">
  <button onclick="loadClip()">Load</button>
  <button onclick="loadRandom()">Random</button>
  <button onclick="toggleList()">List</button>
</div>

<div id="status"></div>
<div id="info" class="info" style="display:none"></div>
<div id="gallery" class="grid"></div>
<div id="clip_list" class="clip-list"></div>

<script>
// Use relative URLs so it works behind any reverse proxy prefix
function apiUrl(path) {
  // Get the base path from the current page URL (handles proxy prefixes)
  var base = window.location.pathname.replace(/\/$/, '');
  return base + path;
}

function loadClip() {
  var cid = document.getElementById('clip_id').value.trim();
  var t0 = document.getElementById('t0_us').value;
  if (!cid) return;

  document.getElementById('status').innerHTML = '<div class="loading">Loading clip...</div>';
  document.getElementById('gallery').innerHTML = '';
  document.getElementById('info').style.display = 'none';

  fetch(apiUrl('/api/clip?clip_id=' + encodeURIComponent(cid) + '&t0_us=' + t0))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      document.getElementById('status').innerHTML = '';
      if (data.error) {
        document.getElementById('status').innerHTML = '<div class="error">' + data.error + '</div>';
        return;
      }
      document.getElementById('info').innerText = data.info;
      document.getElementById('info').style.display = 'block';
      var html = '';
      for (var i = 0; i < data.images.length; i++) {
        html += '<div class="cell">';
        html += '<img src="data:image/jpeg;base64,' + data.images[i].b64 + '">';
        html += '<div class="label">' + data.images[i].label + '</div>';
        html += '</div>';
      }
      document.getElementById('gallery').innerHTML = html;
    })
    .catch(function(e) {
      document.getElementById('status').innerHTML = '<div class="error">Fetch error: ' + e + '</div>';
    });
}

function loadRandom() {
  var t0 = document.getElementById('t0_us').value;
  document.getElementById('status').innerHTML = '<div class="loading">Loading random clip...</div>';
  document.getElementById('gallery').innerHTML = '';
  document.getElementById('info').style.display = 'none';

  fetch(apiUrl('/api/random?t0_us=' + t0))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) {
        document.getElementById('status').innerHTML = '<div class="error">' + data.error + '</div>';
        return;
      }
      document.getElementById('clip_id').value = data.clip_id;
      document.getElementById('status').innerHTML = '';
      document.getElementById('info').innerText = data.info;
      document.getElementById('info').style.display = 'block';
      var html = '';
      for (var i = 0; i < data.images.length; i++) {
        html += '<div class="cell">';
        html += '<img src="data:image/jpeg;base64,' + data.images[i].b64 + '">';
        html += '<div class="label">' + data.images[i].label + '</div>';
        html += '</div>';
      }
      document.getElementById('gallery').innerHTML = html;
    })
    .catch(function(e) {
      document.getElementById('status').innerHTML = '<div class="error">Fetch error: ' + e + '</div>';
    });
}

function toggleList() {
  var el = document.getElementById('clip_list');
  if (el.style.display !== 'none') {
    el.style.display = 'none';
    return;
  }
  el.innerHTML = '<div class="loading">Loading list...</div>';
  el.style.display = 'block';

  fetch(apiUrl('/api/list'))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var html = '';
      for (var i = 0; i < data.clips.length; i++) {
        html += '<span onclick="pickClip(\\''+data.clips[i]+'\\')">'+i+': '+data.clips[i]+'</span>';
      }
      el.innerHTML = html;
    });
}

function pickClip(cid) {
  document.getElementById('clip_id').value = cid;
  document.getElementById('clip_list').style.display = 'none';
  loadClip();
}

document.getElementById('clip_id').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') loadClip();
});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Web clip browser")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/api/clip")
    def api_clip():
        clip_id = request.args.get("clip_id", "").strip()
        t0_us = int(request.args.get("t0_us", 5100000))

        if not clip_id:
            return jsonify({"error": "No clip_id provided"})

        _init_dataset()
        if clip_id not in _avdi.clip_index.index:
            return jsonify({"error": f"Unknown clip_id: {clip_id}"})

        try:
            images, n_cam, n_fr = load_clip_frames(clip_id, t0_us)
            info = (f"clip_id: {clip_id} | t0_us: {t0_us} | "
                    f"Cameras: {n_cam} | Frames/cam: {n_fr}")
            return jsonify({
                "clip_id": clip_id,
                "info": info,
                "images": [{"label": im["label"], "b64": im["b64"]} for im in images],
            })
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/api/random")
    def api_random():
        t0_us = int(request.args.get("t0_us", 5100000))
        _init_dataset()
        clip_id = _valid_clips.sample(1).index[0]
        try:
            images, n_cam, n_fr = load_clip_frames(clip_id, t0_us)
            info = (f"clip_id: {clip_id} | t0_us: {t0_us} | "
                    f"Cameras: {n_cam} | Frames/cam: {n_fr}")
            return jsonify({
                "clip_id": clip_id,
                "info": info,
                "images": [{"label": im["label"], "b64": im["b64"]} for im in images],
            })
        except Exception as e:
            return jsonify({"clip_id": clip_id, "error": str(e)})

    @app.route("/api/list")
    def api_list():
        _init_dataset()
        return jsonify({"clips": list(_valid_clips.index[:100])})

    print(f"Starting clip browser on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
