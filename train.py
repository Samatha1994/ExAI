from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import legacy

def compile_and_train(model, train_dataset, validation_dataset, epochs):
    # adam = Adam(learning_rate=0.001, decay=0.001/30) --as this was giving some error
    # ValueError: decay is deprecated in the new Keras optimizer, please check the docstring for valid arguments, 
    # or use the legacy optimizer, e.g., tf.keras.optimizers.legacy.Adam.
    adam = legacy.Adam(learning_rate=0.001, decay=0.001/30)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_dataset,
        steps_per_epoch=train_dataset.samples // train_dataset.batch_size,
        validation_data=validation_dataset,
        validation_steps=validation_dataset.samples // validation_dataset.batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    return history
