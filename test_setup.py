#!/usr/bin/env python3
"""
Quick test to make sure everything is set up correctly
"""

import sys

def check_imports():
    print("Checking required packages...")
    required = ['pandas', 'numpy', 'sklearn', 'streamlit', 'matplotlib', 'seaborn', 'joblib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed!")
    return True

def check_files():
    print("\nChecking project files...")
    import os
    
    required_files = [
        'app.py',
        'train_model.py', 
        'generate_data.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    
    print("\n✓ All files present!")
    return True

def main():
    print("=" * 50)
    print("Car Price Predictor - Setup Verification")
    print("=" * 50)
    print()
    
    imports_ok = check_imports()
    files_ok = check_files()
    
    print("\n" + "=" * 50)
    if imports_ok and files_ok:
        print("✓ Setup complete! Next steps:")
        print("  1. python generate_data.py")
        print("  2. python train_model.py")
        print("  3. streamlit run app.py")
    else:
        print("✗ Setup incomplete. Fix the issues above.")
        sys.exit(1)
    print("=" * 50)

if __name__ == '__main__':
    main()
