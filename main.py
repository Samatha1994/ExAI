import data_preparation as dp
import model as md
import train as tr
import os

config_path = 'config.json'
config = dp.load_config(config_path)

train_dataset, validation_dataset = dp.create_data_generators(config)
model = md.create_model(train_dataset.num_classes)
history = tr.compile_and_train(model, train_dataset, validation_dataset, config['epochs'])

# model.save('model_resnet50V2_10classes_retest2023June.h5')

# Directory where you want to save outputs
output_dir = 'outputs'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model
model_save_path = os.path.join(output_dir, 'model_resnet50V2_10classes_retest2023June.h5')
model.save(model_save_path)


