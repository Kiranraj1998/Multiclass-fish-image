import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import time
import json
import shutil
from sklearn.model_selection import train_test_split

# Suppress TensorFlow info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Configuration
BASE_DIR = r"C:\Guvi\my-project\Multiclass fish image\fish-classification"
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

# 2. Data Preparation
def setup_data():
    """Automatically create validation set if missing"""
    train_dir = os.path.join(BASE_DIR, "data", "train")
    val_dir = os.path.join(BASE_DIR, "data", "validation")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data not found at {train_dir}")
    
    # Create validation folders if needed
    if not os.path.exists(val_dir) or len(os.listdir(val_dir)) == 0:
        print("Creating validation set...")
        os.makedirs(val_dir, exist_ok=True)
        
        for class_name in os.listdir(train_dir):
            class_train = os.path.join(train_dir, class_name)
            class_val = os.path.join(val_dir, class_name)
            
            if os.path.isdir(class_train):
                os.makedirs(class_val, exist_ok=True)
                images = [f for f in os.listdir(class_train) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if len(images) > 1:
                    _, val_images = train_test_split(images, test_size=0.2, random_state=42)
                    for img in val_images:
                        shutil.copy(
                            os.path.join(class_train, img),
                            os.path.join(class_val, img),
                    print(f"Copied {len(val_images)} images to {class_val}")
                        )

    return train_dir, val_dir

train_dir, val_dir = setup_data()

# 3. Data Generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 4. Model Architectures
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def build_mobilenet():
    base_model = MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 5. Training Function
def train_and_evaluate(model, name, epochs=15):
    callbacks = [
        EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(BASE_DIR, "models", f"{name}.h5"),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    start = time.time()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    train_time = time.time() - start
    
    # Evaluation
    eval_start = time.time()
    loss, acc = model.evaluate(val_generator)
    inf_time = (time.time() - eval_start) / len(val_generator) * 1000
    
    return {
        'accuracy': acc,
        'training_time': train_time,
        'inference_time': inf_time,
        'params': model.count_params()
    }

# 6. Train Models
print("\n=== Training CNN ===")
cnn_model = build_cnn()
cnn_metrics = train_and_evaluate(cnn_model, "cnn_model")

print("\n=== Training MobileNetV2 ===")
mobilenet_model = build_mobilenet()
mobilenet_metrics = train_and_evaluate(mobilenet_model, "mobilenet_model")

# 7. Save Results
with open(os.path.join(BASE_DIR, "models", "class_names.json"), 'w') as f:
    json.dump(list(train_generator.class_indices.keys()), f)

# Comparison Report
comparison = pd.DataFrame({
    'Model': ['CNN', 'MobileNetV2'],
    'Accuracy': [cnn_metrics['accuracy'], mobilenet_metrics['accuracy']],
    'Training Time (min)': [cnn_metrics['training_time']/60, mobilenet_metrics['training_time']/60],
    'Inference Time (ms)': [cnn_metrics['inference_time'], mobilenet_metrics['inference_time']],
    'Parameters': [cnn_metrics['params'], mobilenet_metrics['params']]
})

comparison.to_markdown(os.path.join(BASE_DIR, "reports", "model_comparison.md"))

print("\n=== Training Complete ===")
print(f"Models saved to: {os.path.join(BASE_DIR, 'models')}")
print(f"Reports saved to: {os.path.join(BASE_DIR, 'reports')}")
print("\nModel Comparison:")
print(comparison)