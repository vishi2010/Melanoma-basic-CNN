import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import os

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Paths - UPDATE THESE!
IMAGE_FOLDER = '/Users/vishishtbhatnagar/Downloads/train'
CSV_FILE = '/Users/vishishtbhatnagar/Downloads/train/ALabel.csv'

# Image settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50

# Training settings
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================================
# 2. LOAD AND PREPARE DATA
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80)

df = pd.read_csv(CSV_FILE)

print(f"\nTotal images: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Count target distribution
print(f"\nTarget Distribution:")
print(df['target'].value_counts())
melanoma_count = df['target'].sum()
benign_count = (df['target'] == 0).sum()
print(f"\nBenign (0): {benign_count}")
print(f"Melanoma (1): {melanoma_count}")
print(f"Class imbalance: {melanoma_count / len(df) * 100:.2f}% melanoma")

# Create full file paths
df['filepath'] = df['image_name'].apply(lambda x: os.path.join(IMAGE_FOLDER, x + '.jpg'))

# Verify files exist
print("\n" + "="*80)
print("VERIFYING FILES")
print("="*80)
df['exists'] = df['filepath'].apply(os.path.exists)
print(f"Files found: {df['exists'].sum()} / {len(df)}")

if df['exists'].sum() == 0:
    print("❌ ERROR: No files found! Check your IMAGE_FOLDER path.")
    exit()

df = df[df['exists']].copy()
df = df.drop(columns=['exists'])

# ============================================================================
# 3. STRATIFIED TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("SPLITTING DATA (STRATIFIED)")
print("="*80)

train_val_df, test_df = train_test_split(
    df, 
    test_size=TEST_SPLIT, 
    stratify=df['target'],
    random_state=42
)
        
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), 
    stratify=train_val_df['target'],
    random_state=42
)

# Convert target to string for binary class_mode
train_df['target'] = train_df['target'].astype(str)
val_df['target'] = val_df['target'].astype(str)
test_df['target'] = test_df['target'].astype(str)

print(f"Train set: {len(train_df)} images ({train_df['target'].value_counts().get('1',0)} melanoma)")
print(f"Val set: {len(val_df)} images ({val_df['target'].value_counts().get('1',0)} melanoma)")
print(f"Test set: {len(test_df)} images ({test_df['target'].value_counts().get('1',0)} melanoma)")

# ============================================================================
# 4. COMPUTE CLASS WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("COMPUTING CLASS WEIGHTS")
print("="*80)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['target']),
    y=train_df['target']
)
class_weight_dict = {int(k): v for k, v in zip(np.unique(train_df['target']), class_weights)}

print(f"Class weights: {class_weight_dict}")
print(f"Melanoma samples weighted {class_weight_dict[1]:.2f}x more than benign")

# ============================================================================
# 5. DATA AUGMENTATION
# ============================================================================

print("\n" + "="*80)
print("SETTING UP DATA AUGMENTATION")
print("="*80)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# ============================================================================
# 6. CREATE DATA GENERATORS
# ============================================================================

print("Creating data generators...")

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='target',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='target',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='target',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("✅ Data generators ready!")

# ============================================================================
# 7. BUILD CNN MODEL
# ============================================================================

print("\n" + "="*80)
print("BUILDING CNN MODEL")
print("="*80)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# ============================================================================
# 8. COMPILE MODEL
# ============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# ============================================================================
# 9. CALLBACKS
# ============================================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_melanoma_model.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# ============================================================================
# 10. TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_df) // BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 11. EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80 + "\n")

model.load_weights('best_melanoma_model.keras')

test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_df['target'].astype(int).values

print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma']))

print("\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)
cm = confusion_matrix(y_true, y_pred)
print(cm)
print(f"\nTrue Negatives (Correct Benign): {cm[0,0]}")
print(f"False Positives (Benign called Melanoma): {cm[0,1]}")
print(f"False Negatives (Melanoma called Benign): {cm[1,0]}")
print(f"True Positives (Correct Melanoma): {cm[1,1]}")

print(f"\nROC-AUC Score: {roc_auc_score(y_true, y_pred_probs):.4f}")

# ============================================================================
# 12. VISUALIZE TRAINING HISTORY
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Validation')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Validation')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Validation')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('melanoma_training_history.png', dpi=300, bbox_inches='tight')
print("\n✅ Training history saved to 'melanoma_training_history.png'")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
