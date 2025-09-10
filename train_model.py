import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Force TensorFlow to use CPU
from tensorflow.keras import layers, models, regularizers
import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load synthetic patient data
data = pd.read_csv(r"D:\pneumonia\synthetic_patient_data.csv")
print("Class balance:\n", data['label'].value_counts())

# Define image preprocessing function (move above usage)
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return img_array

# Prepare data before augmentation
image_paths = [os.path.join(r"D:\pneumonia\chest_xray", filename) for filename in data['filename']]
images = np.array([load_and_preprocess_image(path) for path in image_paths])
patient_data = data[['age', 'fever']].values  # Numeric patient features
labels = data['label'].map({'virus': 0, 'bacteria': 1}).values  # Binary labels

# Data augmentation for minority class (Virus)
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
def augment_images(images, labels):
    aug_images = []
    aug_labels = []
    for img, lbl in zip(images, labels):
        if lbl == 0:  # Virus class
            img = np.expand_dims(img, axis=-1)  # Add channel dimension for datagen
            img = np.expand_dims(img, axis=0)   # Add batch dimension
            for _ in range(2):  # Generate 2 augmented images per Virus sample
                aug_img = next(datagen.flow(img, batch_size=1))[0]
                aug_images.append(aug_img[:, :, 0])  # Remove channel dimension
                aug_labels.append(lbl)
    return np.array(aug_images), np.array(aug_labels)

# Save original labels for augmentation indexing
original_labels = labels.copy()
aug_images, aug_labels = augment_images(images[original_labels == 0], original_labels[original_labels == 0])
images = np.concatenate([images, aug_images])
labels = np.concatenate([labels, aug_labels])
patient_data = np.concatenate([patient_data, patient_data[original_labels == 0].repeat(2, axis=0)])

# Split data
X_images_train, X_images_test, X_patient_train, X_patient_test, y_train, y_test = train_test_split(
    images, patient_data, labels, test_size=0.2, random_state=42)

# Define custom lightweight CNN model
image_input = layers.Input(shape=(128, 128, 1))
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dropout(0.6)(x)  # Increase dropout for more regularization

patient_input = layers.Input(shape=(2,))

y = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.05))(patient_input)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.6)(y)

combined = layers.concatenate([x, y])
z = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.05))(combined)
z = layers.Dropout(0.6)(z)
output = layers.Dense(1, activation='sigmoid')(z)

model = models.Model(inputs=[image_input, patient_input], outputs=output)

# Compile with learning rate scheduler
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Compute and print class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)


# Train the model
history = model.fit(
    [X_images_train, X_patient_train],
    y_train,
    epochs=25,
    batch_size=4,
    validation_split=0.2,
    verbose=2,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# Save the trained model
model.save(r"D:\pneumonia\saved_cnn_model.keras")
print("Model saved to D:/pneumonia/saved_cnn_model.keras")

# Evaluate
loss, accuracy = model.evaluate([X_images_test, X_patient_test], y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Predict and other outputs (same as before)
y_pred_prob = model.predict([X_images_test, X_patient_test])
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Virus', 'Bacteria'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

total_cases = np.bincount(y_test, minlength=2)
correct_predictions = np.diag(cm)
performance_data = pd.DataFrame({
    'Disease': ['Virus', 'Bacteria'],
    'Total Cases': total_cases,
    'Correct Predictions': correct_predictions
})
print("\nPerformance Table:")
print(performance_data)

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

def predict_and_get_precautions(image_path, age, fever):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    patient_data = np.array([[age, fever]])
    prediction = model.predict([img, patient_data])
    condition = "bacteria" if prediction[0][0] > 0.5 else "virus"
    precautions = []
    if condition == "virus":
        precautions.append("Rest and stay hydrated to support recovery from viral pneumonia.")
        precautions.append("Consult a doctor if symptoms worsen (e.g., difficulty breathing).")
        if fever == 1: precautions.append("Use fever-reducing medication as advised.")
        if age > 60: precautions.append("Seek immediate medical attention due to age risk.")
    elif condition == "bacteria":
        precautions.append("Seek antibiotic treatment from a healthcare provider.")
        precautions.append("Avoid spreading infection by isolating if possible.")
        if fever == 1: precautions.append("Monitor fever closely and use medication under guidance.")
        if age > 60: precautions.append("Urgent medical evaluation recommended.")
    return condition, precautions

test_image_path = os.path.join(r"D:\pneumonia\chest_xray", data['filename'].iloc[0])
age, fever = data['age'].iloc[0], data['fever'].iloc[0]
condition, precautions = predict_and_get_precautions(test_image_path, age, fever)

print(f"\nPredicted condition for test image: {condition}")
print("Precautions:")
for precaution in precautions:
    print(f"- {precaution}")