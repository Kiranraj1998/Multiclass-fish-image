import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# FIXED PATH HANDLING (Option 1 - Raw String)
base_path = r"C:\Guvi\my-project\Multiclass fish image\fish-classification"
train_path = os.path.join(base_path, "data", "train")
val_path = os.path.join(base_path, "data", "validation")

def verify_data_paths():
    if not os.path.exists(train_path):
        print(f"❌ Error: Missing folder - {train_path}")
        print("Please ensure you:")
        print("1. Created the folder structure")
        print("2. Downloaded the dataset")
        print("3. Extracted images to correct folders")
        return False
        
    if not os.listdir(train_path):
        print(f"❌ Error: Empty folder - {train_path}")
        print(f"Add class subfolders with images to: {train_path}")
        return False
        
    return True

if not verify_data_paths():
    exit(1)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"Found {train_generator.samples} images in {len(train_generator.class_indices)} classes")