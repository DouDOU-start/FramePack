"""
Unified entrypoint for the Gradio demo.

This thin wrapper imports `demo_gradio.py` and executes it.
Keeping logic in `demo_gradio.py` preserves existing functionality
while providing a stable `main.py` entry name.
"""

import demo_gradio  # noqa: F401  (side-effect: builds and launches the app)


