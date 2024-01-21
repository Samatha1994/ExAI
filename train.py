from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import legacy
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report


def compile_and_train(model, train_dataset, validation_dataset, epochs):
    # adam = Adam(learning_rate=0.001, decay=0.001/30) --as this was giving some error
    # ValueError: decay is deprecated in the new Keras optimizer, please check the docstring for valid arguments, 
    # or use the legacy optimizer, e.g., tf.keras.optimizers.legacy.Adam.
    adam = legacy.Adam(learning_rate=0.001, decay=0.001/30)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("[INFO] training model...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_dataset.samples // train_dataset.batch_size,
        validation_data=validation_dataset,
        validation_steps=validation_dataset.samples // validation_dataset.batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    return history

def predict_with_model(model, test_dataset, output_dir):
    predIdxs = model.predict_generator(test_dataset, steps=(test_dataset.samples // test_dataset.batch_size) + 1)
    print(predIdxs.shape)
    print(predIdxs)
    classes = list(np.argmax(predIdxs, axis=1))
    filenames = test_dataset.filenames

    # Save prediction results to DataFrame and CSV
    df = pd.DataFrame(predIdxs)
    df['filenames'] = filenames
    print(df)
    predictions_file = os.path.join(output_dir, 'predictions_of_tenNeurons_dataframe_retest2023June.csv')
    df.to_csv(predictions_file, index=None, header=True)

    # Mapping of classes
    label_map = test_dataset.class_indices
    print(label_map)

    pred_labels = np.argmax(predIdxs, axis=1)

    # Saving class mappings
    b = [(x[0], y) for x in zip(classes, filenames) for y in x[1:]]
    df_labels = pd.DataFrame(b, columns=['Predicted_classes', 'image_name'])
    print (df)
    class_mapping_file = os.path.join(output_dir, 'finalClass_with_image_name_retest2023June.csv')
    df_labels.to_csv(class_mapping_file, index=None, header=True)

    # Return predictions and labels for further evaluation
    return predIdxs, pred_labels, label_map

def evaluate_model(test_dataset, pred_labels):
    print("[INFO] evaluating network...")
    print(classification_report(test_dataset.classes, pred_labels, target_names=test_dataset.class_indices.keys()))

def predict_with_feature_map_model(feature_map_model, test_dataset, output_dir):
    predIdxs = feature_map_model.predict_generator(test_dataset, steps=(test_dataset.samples // test_dataset.batch_size) + 1)
    
    # Assuming 'filenames' is a list of filenames in the same order as test_dataset
    filenames = test_dataset.filenames
    
    df = pd.DataFrame(predIdxs)
    df['filenames'] = filenames
    print(df)
    
    predictions_file = os.path.join(output_dir, 'preds_of_64Neurons_denseLayer_1370Images_retest2023June.csv')
    df.to_csv(predictions_file, index=None, header=True)
    return predIdxs

