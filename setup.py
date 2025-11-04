
import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
import random


def download_lfw_dataset():
    """Download LFW dataset"""
    print("📥 Downloading LFW Dataset...")
    url = "https://www.kaggle.com/datasets/jessicali9530/"
    filename = "lfw_deepfunneled"
    
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename, reporthook=download_progress)
        print("\n✅ Download complete!")
    else:
        print("✅ Dataset already downloaded!")
    
    return filename

def download_progress(block_num, block_size, total_size):
    """Show download progress"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    print(f"\rProgress: {percent:.1f}%", end='')

def extract_dataset(filename):
    """Extract the dataset"""
    print("\n📦 Extracting dataset...")
    
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    print("✅ Extraction complete!")
    return "lfw"

def organize_for_training(source_dir='lfw', target_dir='dataset', min_images=5, max_people=50):
    print(f"\n🗂️  Organizing dataset...")
    
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    people_with_images = {}
    
    for person_name in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) >= min_images:
            people_with_images[person_name] = images
    
    print(f"📊 Found {len(people_with_images)} people with {min_images}+ images")
    
    selected_people = dict(list(people_with_images.items())[:max_people])
    print(f"📌 Selected {len(selected_people)} people for training")
    
    total_train = 0
    total_val = 0
    
    for person_name, images in selected_people.items():
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        train_person_dir = os.path.join(train_dir, person_name)
        val_person_dir = os.path.join(val_dir, person_name)
        
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(val_person_dir, exist_ok=True)
        
        for img in train_images:
            src = os.path.join(source_dir, person_name, img)
            dst = os.path.join(train_person_dir, img)
            shutil.copy2(src, dst)
            total_train += 1
        
        for img in val_images:
            src = os.path.join(source_dir, person_name, img)
            dst = os.path.join(val_person_dir, img)
            shutil.copy2(src, dst)
            total_val += 1
        
        print(f"  ✓ {person_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\n✅ Dataset organized!")
    print(f"   📁 Train: {total_train} images in {train_dir}")
    print(f"   📁 Val: {total_val} images in {val_dir}")
    print(f"   👥 People: {len(selected_people)}")

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 FACE RECOGNITION HACKATHON - QUICK SETUP")
    print("="*60)
    
    filename = download_lfw_dataset()
    
    source_dir = extract_dataset(filename)
    
    organize_for_training(
        source_dir=source_dir,
        target_dir='dataset',
        min_images=5,     
        max_people=50      
    )
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next steps:")
    print("   1. Run your Flask backend: python app.py")
    print("   2. Open the frontend and click 'Train Model'")
    print("   3. Start recognizing faces!")
    print("\n💡 Tip: Training will take 2-5 minutes depending on dataset size")

if __name__ == "__main__":
    main()