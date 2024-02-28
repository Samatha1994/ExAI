# import pandas as pd
# import os

# def generate_config_files(positive_csv, negative_csv, template_config, output_dir, base_url):
#     positive_images = pd.read_csv(positive_csv)
#     negative_images = pd.read_csv(negative_csv)

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     with open(template_config, 'r') as file:
#         template_content = file.read()

#     for col in positive_images.columns:
#         # positive_urls = ['"{}{}"'.format(base_url, os.path.basename(img)) for img in positive_images[col].dropna()]
#         # negative_urls = ['"{}{}"'.format(base_url, os.path.basename(img)) for img in negative_images[col].dropna()]
#         positive_urls = ['"{}{}"'.format(base_url, os.path.splitext(os.path.basename(img))[0]) for img in positive_images[col].dropna()]
#         negative_urls = ['"{}{}"'.format(base_url, os.path.splitext(os.path.basename(img))[0]) for img in negative_images[col].dropna()]

#         config_content = template_content
#         config_content += "\nlp.positiveExamples = {" + ",".join(positive_urls) + "}\n"
#         config_content += "lp.negativeExamples = {" + ",".join(negative_urls) + "}\n"

#         config_filename = f"neuron_{col}.config"
#         with open(os.path.join(output_dir, config_filename), 'w') as output_file:
#             output_file.write(config_content)
import pandas as pd
import os

def generate_config_files(positive_csv, negative_csv, template_config, output_dir, base_url):
    positive_images = pd.read_csv(positive_csv)
    negative_images = pd.read_csv(negative_csv)
    non_empty_config_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(template_config, 'r') as file:
        template_content = file.read()

    for col in positive_images.columns:
        positive_urls = ['"{}{}"'.format(base_url, os.path.splitext(os.path.basename(img))[0]) for img in positive_images[col].dropna()]
        negative_urls = ['"{}{}"'.format(base_url, os.path.splitext(os.path.basename(img))[0]) for img in negative_images[col].dropna()]

        # Skip creating config files for neurons with no positive or negative URLs
        if not positive_urls and not negative_urls:
            print(col+" column is empty")
            continue

        config_content = template_content
        config_content += "\nlp.positiveExamples = {" + ",".join(positive_urls) + "}\n"
        config_content += "lp.negativeExamples = {" + ",".join(negative_urls) + "}\n"

        if positive_urls or negative_urls:
            config_filename = f"neuron_{col}_config.config"
            with open(os.path.join(output_dir, config_filename), 'w') as output_file:
                output_file.write(config_content)
            non_empty_config_count += 1

    print(f"Non-empty config files created: {non_empty_config_count}")
    return non_empty_config_count
