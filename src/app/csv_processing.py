import csv
import json
import os

def csv_to_json(csv_file_path, json_file_path):
    """Convert CSV file to JSON file."""
    try:
        data = []
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
        
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def create_sample_csv(file_path):
    """Create a sample CSV file for testing."""
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Age', 'City'])
        writer.writerow(['Alice', '30', 'New York'])
        writer.writerow(['Bob', '25', 'Los Angeles'])
        writer.writerow(['Charlie', '35', 'Chicago'])