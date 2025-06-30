import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from sklearn.metrics import classification_report, confusion_matrix

# 1. Setup Paths
BASE_DIR = r"C:\Guvi\my-project\Multiclass fish image\fish-classification"
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

# 2. Load Class Names
with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
    class_names = json.load(f)

# 3. Load Test Data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 4. Evaluation Function
def evaluate_model(model_path):
    """Evaluate a single model and return metrics"""
    model = tf.keras.models.load_model(model_path)
    
    # Time inference
    start_time = time.time()
    y_pred = model.predict(test_generator)
    inference_time = (time.time() - start_time) / len(test_generator) * 1000  # ms per image
    
    y_true = test_generator.classes
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred_classes)
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'inference_time_ms': inference_time,
        'confusion_matrix': cm
    }

# 5. Evaluate Both Models
print("\n=== Evaluating Models ===")
metrics = {}

for model_name in ['cnn_model', 'mobilenet_model']:
    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    if os.path.exists(model_path):
        print(f"\nEvaluating {model_name}...")
        metrics[model_name] = evaluate_model(model_path)

# 6. Generate Reports
if metrics:
    # Text Report
    report_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    with open(os.path.join(report_dir, "evaluation_report.md"), "w") as f:
        f.write("# Model Evaluation Report\n\n")
        
        for model_name, result in metrics.items():
            f.write(f"## {model_name}\n")
            f.write(f"- **Accuracy**: {result['accuracy']:.2%}\n")
            f.write(f"- **Precision**: {result['precision']:.2%}\n")
            f.write(f"- **Recall**: {result['recall']:.2%}\n")
            f.write(f"- **F1-Score**: {result['f1']:.2%}\n")
            f.write(f"- **Inference Time**: {result['inference_time_ms']:.2f} ms/image\n\n")
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(12, 10))
    for i, (model_name, result) in enumerate(metrics.items(), 1):
        plt.subplot(1, len(metrics), i)
        sns.heatmap(result['confusion_matrix'], 
                    annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f"{model_name}\nAccuracy: {result['accuracy']:.2%}")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "confusion_matrices.png"))
    plt.close()
    
    # Comparison Table
    comparison_df = pd.DataFrame.from_dict(metrics, orient='index')
    comparison_df.to_markdown(os.path.join(report_dir, "model_comparison.md"))
    
    print("\n=== Evaluation Complete ===")
    print(f"Reports saved to {report_dir}/")
    print(comparison_df[['accuracy', 'inference_time_ms']])
else:
    print("No models found for evaluation!")