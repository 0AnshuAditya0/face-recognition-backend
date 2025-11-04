#!/usr/bin/env python3
"""
Convenient launcher script for Face Recognition System
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import torch
        import facenet_pytorch
        import cv2
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\nInstall requirements:")
        print("  pip install -r requirements.txt")
        return False

def check_dataset():
    """Check if LFW dataset exists"""
    dataset_dir = Path("dataset/lfw-deepfunneled")
    if not dataset_dir.exists():
        print("⚠️  LFW dataset not found!")
        print("\nRun setup:")
        print("  python quick_dataset_setup.py")
        return False
    
    people_count = len([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"✅ Dataset found: {people_count} people")
    return True

def check_model():
    """Check if trained model exists"""
    if Path("face_encodings.pkl").exists():
        print("✅ Trained model found")
        return True
    else:
        print("ℹ️  No trained model (will train on first use)")
        return False

def run_backend():
    """Start FastAPI backend"""
    print("\n" + "="*70)
    print("Starting Backend (FastAPI + PyTorch)")
    print("="*70)
    
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Backend stopped")

def run_frontend():
    """Start Next.js frontend"""
    print("\n" + "="*70)
    print("Starting Frontend (Next.js)")
    print("="*70)
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("\nInstalling frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir)
    
    try:
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)
    except KeyboardInterrupt:
        print("\n\n✅ Frontend stopped")

def run_both():
    """Run both backend and frontend"""
    import threading
    
    def backend_thread():
        run_backend()
    
    # Start backend in thread
    backend = threading.Thread(target=backend_thread, daemon=True)
    backend.start()
    
    print("\n⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend in main thread
    run_frontend()

def show_menu():
    """Show interactive menu"""
    print("\n" + "="*70)
    print("🎭 Face Recognition System Launcher")
    print("="*70)
    print()
    
    # System checks
    print("System Status:")
    print("-" * 70)
    
    if not check_requirements():
        return False
    
    check_dataset()
    check_model()
    
    print("-" * 70)
    print()
    
    # Menu options
    print("Options:")
    print("  1. Setup Dataset (download LFW)")
    print("  2. Run Backend Only (FastAPI)")
    print("  3. Run Frontend Only (Next.js)")
    print("  4. Run Full Stack (Backend + Frontend)")
    print("  5. Check System Status")
    print("  0. Exit")
    print()
    
    return True

def main():
    """Main launcher"""
    if not show_menu():
        return
    
    try:
        choice = input("Enter choice (0-5): ").strip()
        
        if choice == '1':
            print("\n📥 Starting dataset setup...")
            subprocess.run([sys.executable, "quick_dataset_setup.py"])
            
        elif choice == '2':
            run_backend()
            
        elif choice == '3':
            run_frontend()
            
        elif choice == '4':
            print("\n🚀 Starting full stack...")
            print("Backend: http://127.0.0.1:5000")
            print("Frontend: http://localhost:3000")
            print("\nPress Ctrl+C to stop\n")
            run_both()
            
        elif choice == '5':
            show_menu()
            
        elif choice == '0':
            print("\n👋 Goodbye!")
            
        else:
            print("\n❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()