import csv
import json
from tabulate import tabulate
from ml_model import FieldMappingModel
from fuzzywuzzy import process  # Import fuzzy matching library
import openai  # Import OpenAI library
import os

# Load the mapping configuration
with open('/Users/samohan/Code/mapping/ServiceNowToFreshservice.json') as f:
    mapping_config = json.load(f)

# Load the default mapping
with open('/Users/samohan/Code/mapping/ServiceNowToFreshserviceDefaultMapping.json') as f:
    default_mapping = json.load(f)

# Initialize the field mapping model
model = FieldMappingModel()
model.load_data('/Users/samohan/Code/mapping/field_mappings.json')
model.train()

# Read OpenAI API key from a local file
key_file_path = '/Users/samohan/Code/ai-copilot/openai_key.txt'
if os.path.exists(key_file_path):
    with open(key_file_path, 'r') as key_file:
        openai.api_key = key_file.read().strip()
else:
    raise FileNotFoundError(f"OpenAI API key file not found at {key_file_path}")

def load_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader)

def map_fields(servicenow_data, mapping_config, default_mapping):
    mapped_data = []
    unmapped_fields = set()
    mapped_fields = set()
    field_summary = {}

    # Create a dictionary for quick lookup of default mappings
    default_mapping_dict = {item['sourcefield']: item['targetfield'] for item in default_mapping['fieldMappings']}

    for record in servicenow_data:
        mapped_record = {}
        for target in mapping_config['additional_details']['targets']:
            for field_mapping in target['fieldMappings']:
                source_field = field_mapping.get('sourcefield')
                target_field = field_mapping.get('targetfield')
                if source_field in record:
                    mapped_record[target_field] = record[source_field]
                    mapped_fields.add(source_field)
                    field_summary[source_field] = target_field
                elif source_field in default_mapping_dict:
                    mapped_record[target_field] = default_mapping_dict[source_field]
                    mapped_fields.add(source_field)
                    field_summary[source_field] = default_mapping_dict[source_field]
                else:
                    unmapped_fields.add(source_field)
        mapped_data.append(mapped_record)
    
    return mapped_data, unmapped_fields, mapped_fields, field_summary

def suggest_possible_conversion(field):
    try:
        # Use the trained model for prediction
        return model.predict(field)
    except:
        # Fallback to GPT-4 for generating suggestions
        try:
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=f"Suggest a possible mapping for the field '{field}' in the context of SaaS applications.",
                max_tokens=50,
                temperature=0.7
            )
            suggestion = response.choices[0].text.strip()
            return suggestion if suggestion else "No suggestion available"
        except Exception as e:
            print(f"Error using GPT-4: {e}")
            return "No suggestion available"

def query_user_for_suggestions(unmapped_fields):
    for field in unmapped_fields:
        suggestion = suggest_possible_conversion(field)
        print(f"Suggested mapping for '{field}': {suggestion}")
        user_input = input(f"Please provide a target field for unmapped field '{field}' (or press Enter to accept suggestion): ")
        if user_input:
            model.update(field, user_input)
        else:
            model.update(field, suggestion)

    # Save updated model data
    model.save_data('/Users/samohan/Code/mapping/field_mappings.json')

def print_unmapped_fields(unmapped_fields):
    table = [[field, suggest_possible_conversion(field)] for field in unmapped_fields]
    print(tabulate(table, headers=["Unmapped Field", "Possible Conversion"], tablefmt="grid"))

def print_field_summary(field_summary):
    table = [[source, target] for source, target in field_summary.items()]
    print(tabulate(table, headers=["Source Field", "Target Field"], tablefmt="grid"))

def main():
    servicenow_data = load_csv('/Users/samohan/Code/mapping/75.csv')
    _, unmapped_fields, _, field_summary = map_fields(servicenow_data, mapping_config, default_mapping)
    
    # Print unmapped fields
    print_unmapped_fields(unmapped_fields)
    
    # Query user for suggestions on unmapped fields
    query_user_for_suggestions(unmapped_fields)
    
    # Re-map fields with updated user suggestions
    mapped_data, unmapped_fields, mapped_fields, field_summary = map_fields(servicenow_data, mapping_config, default_mapping)
    
    # Print field summary
    print_field_summary(field_summary)

if __name__ == "__main__":
    main()
