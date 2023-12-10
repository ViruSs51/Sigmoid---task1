import csv
import json

class File:

    def get_csv(self, 
                file: str
                ) -> list:
        with open(file, newline='') as csv_file:
            data_file = [row for row in csv.reader(csv_file, delimiter=',', quotechar='|')][1:]

        return data_file

    def get_json(self, 
                 file: str
                 ) -> dict:
        with open(file, 'r') as json_file:
            data_file = json.loads(json_file.read())
        
        return data_file
    
    def put_json(self, 
                 file: str, 
                 data: dict
                 ) -> None:
        with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=4)