import data_preparation as dp
import model as md
import train as tr
import os
import analyze_activations as aa
import config_generator as cg
import download_from_gdrive as dgdrive


config_path = 'config.json'
config = dp.load_config(config_path)

train_dataset, test_dataset, validation_dataset = dp.create_data_generators(config)
model = md.create_model(train_dataset.num_classes)
# -----------------------------------------------------------------------------------------
#UNCOMMENT later:  commenting below to avoid repeating to compile and train again
history = tr.compile_and_train(model, train_dataset, validation_dataset, config['epochs'])
# ------------------------------------------------------------------------------------------
# Directory where you want to save outputs
output_dir = 'outputs'
model_filename = 'model_resnet50V2_10classes_retest2023June.h5'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model
model_save_path = os.path.join(output_dir, model_filename)
# ---------------------------------------------------------------------------------------------
#UNCOMMENT later: Reusing exiting .h5 file..commenting below to avoid repeating to save the .h5 file.
model.save(model_save_path)
# -----------------------------------------------------------------------------------------------
model, layer_outputs, layer_names, feature_map_model = md.load_and_analyze_model(model_save_path)


predIdxs, pred_labels, label_map = tr.predict_with_model(model, test_dataset, output_dir)
tr.evaluate_model(test_dataset, pred_labels)
predIdxs= tr.predict_with_feature_map_model(feature_map_model, test_dataset, output_dir)


csv_file_path = 'outputs/preds_of_64Neurons_denseLayer_1370Images_retest2023June.csv'
# Analyze activations and save results
aa.analyze_activations(csv_file_path, output_dir)



# Define your file paths and parameters
output_dir = './outputs/config_files'
# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
positive_csv_path = './outputs/positive_images.csv'
negative_csv_path = './outputs/negative_images.csv'
template_config_path = './inputs/set6-initial_score_hybrid.config'
base_url = "http://www.daselab.org/ontologies/ADE20K/hcbdwsu#"

# Call the function to generate config files
###cg.generate_config_files(positive_csv_path, negative_csv_path, template_config_path, output_dir, base_url)
# Generate config files and get the count of non-empty config files created
non_empty_config_count = cg.generate_config_files(positive_csv_path, negative_csv_path, template_config_path, output_dir, base_url)

print(f"Total non-empty config files created: {non_empty_config_count}")

#-------------------------------------------------------------------------------------------
#pip install gdown
# DOWNLOAD .owl file from google drive

# URL of the shared Google Drive file and output folder
# file_url = 'https://drive.google.com/file/d/1JlaelTKQczDMsm6aAlQVSw8gQqDnybSD/view?usp=sharing'
# output_folder = 'outputs'
# filename = 'combined.owl'  # Set your desired filename here

# Download the file
#UNCOMMENT later:  
# file_path = dgdrive.download_file_from_google_drive(file_url, output_folder, filename)
# print(f"File downloaded to: {file_path}")


#------------------------------------------------------------------------------------------


