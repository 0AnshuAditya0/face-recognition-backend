import os
import shutil
from sklearn.model_selection import train_test_split
import random

def organize_lfw_dataset(source_dir='lfw', target_dir='dataset', min_images=10):
    """
    Organize LFW dataset into train/val split
    Only keeps people with at least min_images photos
    """
    
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print("Organizing dataset...")
    
    # Get all person folders
    people = {}
    for person_name in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) >= min_images:
            people[person_name] = images
    
    print(f"Found {len(people)} people with at least {min_images} images")
    
    # Split each person's images 80/20
    for person_name, images in people.items():
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create person folders
        train_person_dir = os.path.join(train_dir, person_name)
        val_person_dir = os.path.join(val_dir, person_name)
        
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(val_person_dir, exist_ok=True)
        
        # Copy training images
        for img in train_images:
            src = os.path.join(source_dir, person_name, img)
            dst = os.path.join(train_person_dir, img)
            shutil.copy2(src, dst)
        
        # Copy validation images
        for img in val_images:
            src = os.path.join(source_dir, person_name, img)
            dst = os.path.join(val_person_dir, img)
            shutil.copy2(src, dst)
        
        print(f"✓ {person_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\n✅ Dataset organized!")
    print(f"Train: {train_dir}")
    print(f"Val: {val_dir}")

if __name__ == "__main__":
    # Adjust these parameters based on your needs
    organize_lfw_dataset(
        source_dir='lfw',  # Your downloaded LFW folder
        target_dir='dataset',
        min_images=10  # Minimum images per person
    )