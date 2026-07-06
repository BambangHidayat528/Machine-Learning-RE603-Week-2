# ============================================================================
# IMAGE CLASSIFICATION: DETEKSI JENIS SAMPAH DENGAN TRANSFER LEARNING
# Dataset: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. LOAD DAN PREPROCESSING DATA
# ============================================================================

# Konfigurasi
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'garbage_classification'  # Folder dataset setelah di-download dan di-extract

# Load dataset
print("Loading dataset...")
train_ds = image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=42,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42,
    shuffle=True
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")

# Prefetch untuk performa
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================================================
# 2. DATA AUGMENTATION (Teknik 1: Mencegah Overfitting)
# ============================================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

# ============================================================================
# 3. TRANSFER LEARNING DENGAN MOBILENETV2
# ============================================================================

# Load pretrained MobileNetV2 (lightweight, cocok untuk deployment)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Freeze base model (tidak train ulang layer pretrained)
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)  # Augmentasi data
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Preprocessing khusus MobileNet
x = base_model(x, training=False)  # Base model (frozen)

# Custom classifier head dengan Dropout (Teknik 2: Mencegah Overfitting)
x = layers.Dropout(0.5)(x)  # Dropout 50%
x = layers.Dense(128, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.001))(x)  # L2 Regularization (Teknik 3)
x = layers.Dropout(0.3)(x)  # Dropout lagi
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================================
# 4. CALLBACKS UNTUK MENCEGAH OVERFITTING
# ============================================================================

# Early Stopping (Teknik 4)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce Learning Rate on Plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# ============================================================================
# 5. TRAINING MODEL (FASE 1: FROZEN BASE)
# ============================================================================

print("\n" + "="*60)
print("TRAINING FASE 1: FROZEN BASE MODEL")
print("="*60)

history_frozen = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# ============================================================================
# 6. CEK OVERFITTING
# ============================================================================

print("\n" + "="*60)
print("CEK OVERFITTING")
print("="*60)

train_acc = history_frozen.history['accuracy']
val_acc = history_frozen.history['val_accuracy']
train_loss = history_frozen.history['loss']
val_loss = history_frozen.history['val_loss']

print(f"\nTraining Accuracy (epoch terakhir): {train_acc[-1]:.4f}")
print(f"Validation Accuracy (epoch terakhir): {val_acc[-1]:.4f}")
print(f"Training Loss (epoch terakhir): {train_loss[-1]:.4f}")
print(f"Validation Loss (epoch terakhir): {val_loss[-1]:.4f}")

# Cek overfitting
acc_gap = train_acc[-1] - val_acc[-1]
loss_gap = val_loss[-1] - train_loss[-1]

print(f"\nGap Accuracy (Train - Val): {acc_gap:.4f}")
print(f"Gap Loss (Val - Train): {loss_gap:.4f}")

if acc_gap > 0.10 or loss_gap > 0.5:
    print("\n⚠️  TERDETEKSI OVERFITTING!")
    print("   - Accuracy training jauh lebih tinggi dari validation")
    print("   - Loss validation jauh lebih tinggi dari training")
else:
    print("\n✅ Tidak ada overfitting yang signifikan")

# ============================================================================
# 7. VISUALISASI HASIL TRAINING
# ============================================================================

plt.figure(figsize=(14, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Accuracy', marker='o')
plt.plot(val_acc, label='Val Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss', marker='o')
plt.plot(val_loss, label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history_frozen.png', dpi=300)
plt.show()

# ============================================================================
# 8. FINE-TUNING (FASE 2: UNFREEZE BEBERAPA LAYER)
# ============================================================================

print("\n" + "="*60)
print("TRAINING FASE 2: FINE-TUNING")
print("="*60)

# Unfreeze beberapa layer teratas dari base model
base_model.trainable = True

# Freeze semua layer kecuali 20 layer teratas
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile dengan learning rate lebih kecil
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # LR lebih kecil untuk fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Lanjut training
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    initial_epoch=len(history_frozen.history['loss']),
    callbacks=callbacks
)

# Combine histories
history = {
    'accuracy': history_frozen.history['accuracy'] + history_finetune.history['accuracy'],
    'val_accuracy': history_frozen.history['val_accuracy'] + history_finetune.history['val_accuracy'],
    'loss': history_frozen.history['loss'] + history_finetune.history['loss'],
    'val_loss': history_frozen.history['val_loss'] + history_finetune.history['val_loss']
}

# ============================================================================
# 9. EVALUASI FINAL
# ============================================================================

print("\n" + "="*60)
print("EVALUASI FINAL")
print("="*60)

final_train_acc = history['accuracy'][-1]
final_val_acc = history['val_accuracy'][-1]
final_train_loss = history['loss'][-1]
final_val_loss = history['val_loss'][-1]

print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Cek overfitting setelah fine-tuning
final_acc_gap = final_train_acc - final_val_acc
final_loss_gap = final_val_loss - final_train_loss

print(f"\nFinal Gap Accuracy: {final_acc_gap:.4f}")
print(f"Final Gap Loss: {final_loss_gap:.4f}")

if final_acc_gap > 0.10 or final_loss_gap > 0.5:
    print("\n⚠️  Masih ada indikasi overfitting")
    print("   Saran: Tingkatkan augmentasi data atau dropout")
else:
    print("\n✅ Model generalisasi dengan baik")

# ============================================================================
# 10. VISUALISASI FINAL HISTORY
# ============================================================================

plt.figure(figsize=(14, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Val Accuracy', marker='s')
plt.axvline(x=len(history_frozen.history['loss']), color='red', linestyle='--', 
            label='Fine-tuning Start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch (Full Training)')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Val Loss', marker='s')
plt.axvline(x=len(history_frozen.history['loss']), color='red', linestyle='--',
            label='Fine-tuning Start')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch (Full Training)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history_full.png', dpi=300)
plt.show()

# ============================================================================
# 11. EVALUASI DENGAN CONFUSION MATRIX
# ============================================================================

print("\n" + "="*60)
print("CONFUSION MATRIX & CLASSIFICATION REPORT")
print("="*60)

# Get predictions
y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    y_pred.extend(pred_classes)
    y_true.extend(labels.numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ============================================================================
# 12. SIMPAN MODEL
# ============================================================================

# Simpan model lengkap
model.save('trash_classification_model.h5')
print("\n✅ Model berhasil disimpan: trash_classification_model.h5")

# Simpan dalam format SavedModel (untuk deployment)
model.save('trash_classification_savedmodel')
print("✅ Model berhasil disimpan: trash_classification_savedmodel/")

# ============================================================================
# 13. FUNGSI PREDIKSI (UNTUK TESTING)
# ============================================================================

def predict_trash_type(image_path, model, class_names, img_size=224):
    """
    Fungsi untuk memprediksi jenis sampah dari gambar
    
    Parameters:
    - image_path: path ke file gambar
    - model: model yang sudah di-train
    - class_names: list nama kelas
    - img_size: ukuran input model
    
    Returns:
    - predicted_class: kelas yang diprediksi
    - confidence: tingkat kepercayaan
    """
    # Load dan preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    
    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    # Display
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(f"Prediksi: {predicted_class}\nKonfidensi: {confidence:.2f}%")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence

# Contoh penggunaan:
# predict_trash_type('test_image.jpg', model, class_names)

print("\n" + "="*60)
print("IMPLEMENTASI SELESAI!")
print("="*60)
print("\nTeknik Handling Overfitting yang digunakan:")
print("1. ✅ Data Augmentation (flip, rotation, zoom, translation)")
print("2. ✅ Dropout (50% dan 30%)")
print("3. ✅ L2 Regularization (lambda=0.001)")
print("4. ✅ Early Stopping (patience=5)")
print("5. ✅ Reduce Learning Rate on Plateau")
print("6. ✅ Transfer Learning (pretrained weights)")
print("\nFile output:")
print("- training_history_frozen.png")
print("- training_history_full.png")
print("- confusion_matrix.png")
print("- trash_classification_model.h5")
print("- trash_classification_savedmodel/")