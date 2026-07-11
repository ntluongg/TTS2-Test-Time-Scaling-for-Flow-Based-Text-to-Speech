# TTS² Evaluation Demo

This directory contains the supplementary interactive web demo for evaluating the TTS² (Proposed) model against F5-TTS (Baseline).

## How to Review the Demo

Because the demo dynamically loads evaluation data and metrics from a JSON file, opening `index.html` directly in your browser (via `file://`) will not work due to CORS security restrictions. 

You must run a local web server to view the page correctly.

### Instructions:

1. **Open your terminal** and navigate to this `demo` directory.
2. **Start a local HTTP server** using Python (which comes pre-installed on most systems):
   ```bash
   python3 -m http.server 8000
   ```
   *(If you are using Python 2, use `python -m SimpleHTTPServer 8000`)*
3. **Open your web browser** and navigate to:
   [http://localhost:8000](http://localhost:8000)

## Contents
- `index.html`: The main web page UI.
- `audio/`: Contains the comparison audio files (`.wav`) and the metadata (`web_demo_data.json`).
- `TTS2_method.png`: The diagram for the Model Overview section.
