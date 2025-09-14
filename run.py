#!/usr/bin/env python3

import os
import subprocess
import sys

def check_requirements():
    """Check if required packages are installed."""
    try:
        import paddleocr
        import streamlit
        from modules.ocr_module import OCRModule
        from modules.extraction_module import DataExtractionModule
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def main():
    print("bcc mvp")
    print("=" * 40)

    if not check_requirements():
        sys.exit(1)

    print("\nStarting Streamlit app...")
    print("Access the app at: http://localhost:8501")
    print("\nTo stop the app, press Ctrl+C")
    print("-" * 40)

    try:
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error starting app: {e}")

if __name__ == "__main__":
    main()