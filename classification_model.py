import pandas as pd
import numpy as np
import urllib
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow
import io
from contextlib import redirect_stdout
from sklearn.metrics import classification_report
import json

# -------------------------------------------------------------------------
# MLflow
# -------------------------------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run(run_name="Tate_Collection_Hierarchical_Classification"):

# -------------------------------------------------------------------------
# 1. Ucitavanje i priprema podataka
# -------------------------------------------------------------------------

    file_path = 'D:/venv sa python 3.9/the-tate-collection.csv'
    df = pd.read_csv(file_path, sep=';')

    df.drop(columns=['inscription', 'thumbnailCopyright'], inplace=True, errors='ignore')
    df.dropna(subset=['thumbnailUrl', 'url', 'medium'], inplace=True)

    def extract_first_word(text):
        words = str(text).split(',')
        return words[0].split('.')[0]

    df['medium'] = df['medium'].apply(extract_first_word)

# -------------------------------------------------------------------------
# 2. Filtriranje klasa (zadržavamo one koje imaju >= 50 podataka)
# -------------------------------------------------------------------------

    class_counts = df['medium'].value_counts()
    classes_to_keep = class_counts[class_counts >= 50].index
    classes_to_keep = classes_to_keep[classes_to_keep != "Video"]
    classes_to_remove = class_counts[class_counts < 50].index

    filtered_df = df[df['medium'].isin(classes_to_remove)]
    filtered_df.to_csv('D:/venv sa python 3.9/filtered_tate_collection.csv', index=False)

    df = df[df['medium'].isin(classes_to_keep)]
    print("Classes remaining after filtering:", list(classes_to_keep))


# -------------------------------------------------------------------------
# 3. Definiranje hijerarhijskih oznaka
# -------------------------------------------------------------------------

    category_mapping = {
        "Acrylic paint on canvas": "Watercolour",
        "Aquatint on paper": "Watercolour",
        "Bronze": "Sculpture",
        "Plaster": "Sculpture",        
        "Drypoint on paper": "Engraving",
        "Engraving and etching on paper": "Etching",
        "Engraving on paper": "Engraving",               
        "Intaglio print on paper": "Engraving",
        "Line engraving on paper": "Engraving",
        "Linocut on paper": "Engraving",
        "Lithograph on paper": "Engraving",
        "Mezzotint on paper": "Engraving",        
        "Pastel on paper": "Engraving",
        "Pen and ink and graphite on paper": "Graphite",
        "Pen and ink on paper": "ink",
        "Print on paper": "Print",
        "Screenprint and lithograph on paper": "Print",
        "Screenprint on paper": "Print",
        "Graphite": "Graphite",
        "Graphite and chalk on paper": "Graphite",
        "Graphite and gouache on paper": "Graphite",
        "Graphite and ink on paper": "Graphite",
        "Graphite and watercolour on paper": "Graphite",
        "Graphite on paper": "Graphite",
        "Etching": "Etching",
        "Etching and aquatint on paper": "Etching",
        "Etching and drypoint on paper": "Etching",
        "Etching and engraving on paper": "Etching",
        "Etching and mezzotint on paper": "Etching",
        "Etching and watercolour on paper": "Etching",
        "Etching on paper": "Etching",
        "Oil paint": "Oil",
        "Oil paint on board": "Oil",
        "Oil paint on canvas": "Oil",
        "Oil paint on hardboard": "Oil",
        "Oil paint on mahogany": "Oil",
        "Oil paint on paper": "Oil",
        "Oil paint on wood": "Oil",
        "Ink": "Ink",
        "Ink and graphite on paper": "Ink",
        "Ink and watercolour on paper": "Ink",
        "Ink on paper": "Ink",
        "Ink wash and watercolour on paper": "Ink",
        "Wood": "Wood",
        "Wood engraving on paper": "Wood",
        "Woodcut on paper": "Wood",
        "Watercolour": "Watercolour",
        "Watercolour and gouache on paper": "Watercolour",
        "Watercolour and graphite on paper": "Graphite",
        "Watercolour and ink on paper": "Watercolour",
        "Watercolour on paper": "Watercolour",
        "Digital print on paper": "Print",
        "Photograph": "Photo",
        "Gouache": "Gouache",
        "Gouache and graphite on paper": "Gouache",
        "Gouache and watercolour on paper": "Gouache",
        "Gouache on paper": "Gouache",
        "Chalk": "Chalk",
        "Chalk and graphite on paper": "Chalk",
        "Chalk and watercolour on paper": "Chalk",
        "Chalk on paper": "Chalk",     
       
    }
    df['broad_category'] = df['medium'].map(category_mapping)
    df['hierarchical_label'] = df['broad_category'] + " > " + df['medium']

# -------------------------------------------------------------------------
# 4. Preuzimanje slika loklano (samo jednom)
# -------------------------------------------------------------------------

    output_dir = "D:/venv sa python 3.9/tate_images"
    os.makedirs(output_dir, exist_ok=True)

    def download_image(url, output_dir=output_dir):
        
        try:
            filename = os.path.join(output_dir, os.path.basename(url))
            if not os.path.exists(filename):
                urllib.request.urlretrieve(url, filename)
            return filename
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    df['local_path'] = df['thumbnailUrl'].apply(download_image)
    df.dropna(subset=['local_path'], inplace=True)


    samples_for_csv = []
    for cls in classes_to_keep:
        cls_df = df[df['medium'] == cls]
        if len(cls_df) >= 10:
            sample_cls_df = cls_df.sample(n=10, random_state=42)
        else:
            sample_cls_df = cls_df
        samples_for_csv.append(sample_cls_df)

    sample_csv_df = pd.concat(samples_for_csv)
    sample_csv_path = "D:/venv sa python 3.9/tate_10_samples_each_class.csv"
    sample_csv_df.to_csv(sample_csv_path, index=False)
    print(f"Saved 10-sample CSV to: {sample_csv_path}")

    df = df[~df.index.isin(sample_csv_df.index)]
    hierarchical_label_encoder = LabelEncoder()
    df['encoded_hierarchical_label'] = hierarchical_label_encoder.fit_transform(df['hierarchical_label'])

# -------------------------------------------------------------------------
# 5. Podjela podataka
# -------------------------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df['hierarchical_label']
    )

    print("Train split size:", len(train_df))
    print("Test split size:", len(test_df))

# -------------------------------------------------------------------------
# 6. Stvaranje i pripremanje dataseta
# -------------------------------------------------------------------------

    def parse_image(filename, label):
        """
        Reads the image from disk, decodes it, resizes it, and normalizes pixels.
        """
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [128, 128])
        image = image / 255.0
        return image, label

    def create_dataset(dataframe, batch_size=32, shuffle=True):
        """
        Build a tf.data.Dataset from local file paths and hierarchical labels.
        """
        file_paths = dataframe['local_path'].values
        labels = dataframe['encoded_hierarchical_label'].values

        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds


    
    batch_size = 64
    train_dataset = create_dataset(train_df, batch_size=batch_size, shuffle=True)
    test_dataset = create_dataset(test_df, batch_size=batch_size, shuffle=False)

# -------------------------------------------------------------------------
# 7. Encode hijerarhijskih oznaka
# -------------------------------------------------------------------------

    mlflow.log_param("hierarchical_classes", list(hierarchical_label_encoder.classes_))

# -------------------------------------------------------------------------
# 8. Izrada modela
# -------------------------------------------------------------------------

    num_hierarchical_classes = df['encoded_hierarchical_label'].nunique()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_hierarchical_classes, activation='softmax')
    ])
   

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    summary_buf = io.StringIO()
    with redirect_stdout(summary_buf):
        model.summary()
    model_summary_str = summary_buf.getvalue()
    mlflow.log_text(model_summary_str, "model_summary.txt")
    print(model_summary_str)

# -------------------------------------------------------------------------
# 9. Treniranje modela
# -------------------------------------------------------------------------

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=test_dataset,
        callbacks=[early_stop],
    )
    classes = hierarchical_label_encoder.classes_
    np.save("label_classes.npy", classes)
    mlflow.log_artifact("label_classes.npy")
    mlflow.log_metric("final_training_loss", history.history['loss'][-1])
    mlflow.log_metric("final_validation_loss", history.history['val_loss'][-1])
    mlflow.log_metric("final_training_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("final_validation_accuracy", history.history['val_accuracy'][-1])

# -------------------------------------------------------------------------
# 10. Procjena modela
# -------------------------------------------------------------------------

    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)

# -------------------------------------------------------------------------
# 11. Hijerarhijsko mapiranje oznaka
# -------------------------------------------------------------------------

    hierarchical_mapping = {
        idx: label for idx, label in enumerate(hierarchical_label_encoder.classes_)
    }
    mlflow.log_dict(hierarchical_mapping, "hierarchical_label_mapping.json")

# -------------------------------------------------------------------------
# 12. Spremanje modela
# -------------------------------------------------------------------------

    mlflow.tensorflow.log_model(
        model,
        artifact_path="model",
        registered_model_name="Tate_Hierarchical_Model"
    )

# -------------------------------------------------------------------------
# 13. Primjeri predikcija
# -------------------------------------------------------------------------

    small_test_df = test_df.head(16)
    small_test_ds = create_dataset(small_test_df, batch_size=16, shuffle=False)

    preds = model.predict(small_test_ds)
    predicted_classes = np.argmax(preds, axis=1)
    test_labels = small_test_df['encoded_hierarchical_label'].values

    for i in range(min(5, len(predicted_classes))):
        pred_label = predicted_classes[i]
        true_label = test_labels[i]
        pred_hierarchical_label = hierarchical_label_encoder.inverse_transform([pred_label])[0]
        true_hierarchical_label = hierarchical_label_encoder.inverse_transform([true_label])[0]
        print(f"Sample {i}: Prediction={pred_hierarchical_label}, Truth={true_hierarchical_label}")


# -------------------------------------------------------------------------
# 14. Klasifikacijsko izvješće
# -------------------------------------------------------------------------

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

report = classification_report(
    y_true, 
    y_pred, 
    target_names=hierarchical_label_encoder.classes_, 
    output_dict=True
)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=hierarchical_label_encoder.classes_))

report_path = "classification_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)
mlflow.log_artifact(report_path)