import csv
import json
from tabulate import tabulate
from ml_model import FieldMappingModel
from fuzzywuzzy import process  # Import fuzzy matching library
from openai import OpenAI  # Import OpenAI library
import os
import asyncio  # Import asyncio for handling async functions

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

# Read OpenAI API key from a local file and initialize the client
key_file_path = '/Users/samohan/Code/ai-copilot/openai_key.txt'
if os.path.exists(key_file_path):
    with open(key_file_path, 'r') as key_file:
        client = OpenAI(api_key=key_file.read().strip())
else:
    raise FileNotFoundError(f"OpenAI API key file not found at {key_file_path}")

# Common field mapping rules
FIELD_MAPPING_RULES = {
    # Name-related fields
    'name': 'name',
    'full_name': 'full_name',
    'first_name': 'first_name',
    'last_name': 'last_name',
    'username': 'username',
    'user_name': 'username',
    'display_name': 'display_name',
    
    # Contact information
    'email': 'email',
    'phone': 'phone',
    'mobile': 'mobile_phone',
    'address': 'address',
    'location': 'location',
    
    # Date fields
    'date': 'date',
    'created': 'created_at',
    'updated': 'updated_at',
    'modified': 'modified_at',
    'last_modified': 'last_modified_at',
    
    # Status fields
    'status': 'status',
    'state': 'state',
    'active': 'is_active',
    'enabled': 'is_enabled',
    
    # Description fields
    'description': 'description',
    'details': 'details',
    'notes': 'notes',
    'comment': 'comments',
    
    # ID fields
    'id': 'id',
    'identifier': 'identifier',
    'reference': 'reference_number',
    
    # Priority fields
    'priority': 'priority',
    'severity': 'severity',
    'impact': 'impact',
    
    # Category fields
    'category': 'category',
    'type': 'type',
    'group': 'group',
    
    # Assignment fields
    'assignee': 'assigned_to',
    'owner': 'owned_by',
    'manager': 'managed_by',
    
    # Custom fields
    'custom': 'custom_field',
    'cf_': 'custom_field_'
}

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
    # Use GPT-3.5-turbo with the latest OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in SaaS application field mappings. Your task is to suggest the most appropriate field mapping based on common SaaS field naming conventions. Consider the context of ServiceNow to Freshservice migration."},
                {"role": "user", "content": f"Given a source field named '{field}', suggest the most appropriate target field name that would be used in Freshservice. Consider common naming conventions and field purposes. Return only the suggested field name without any explanation."}
            ],
            max_tokens=30,
            temperature=0.3  # Lower temperature for more consistent results
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion if suggestion else "No suggestion available"
    except Exception as e:
        print(f"Error using GPT-3.5: {e}")
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
