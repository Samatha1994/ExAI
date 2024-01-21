import data_preparation as dp
import model as md
import train as tr
import os

config_path = 'config.json'
config = dp.load_config(config_path)

train_dataset, test_dataset, validation_dataset = dp.create_data_generators(config)
model = md.create_model(train_dataset.num_classes)
# history = tr.compile_and_train(model, train_dataset, validation_dataset, config['epochs'])

# Directory where you want to save outputs
output_dir = 'outputs'
model_filename = 'model_resnet50V2_10classes_retest2023June.h5'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model
model_save_path = os.path.join(output_dir, model_filename)
# model.save(model_save_path)
model, layer_outputs, layer_names, feature_map_model = md.load_and_analyze_model(model_save_path)
#abc

predIdxs, pred_labels, label_map = tr.predict_with_model(model, test_dataset, output_dir)
tr.evaluate_model(test_dataset, pred_labels)
predIdxs= tr.predict_with_feature_map_model(feature_map_model, test_dataset, output_dir)

