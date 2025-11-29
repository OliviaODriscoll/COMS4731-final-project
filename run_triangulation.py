#!/usr/bin/env python3
"""
Quick script to run triangulation on synchronized 2D skeletons.
"""

import subprocess
import sys
from pathlib import Path

def main():
    script_path = Path(__file__).parent / "triangulate_from_sync_videos.py"
    
    cmd = [
        sys.executable,
        str(script_path)
    ]
    
    print("Triangulating 3D skeletons from synchronized 2D views...")
    print()
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

