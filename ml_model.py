import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class FieldMappingModel:
    def __init__(self):
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())
        self.data = []
        self.labels = []

    def load_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.data = data['fields']
                self.labels = data['mappings']
        except FileNotFoundError:
            self.data = []
            self.labels = []

    def save_data(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({'fields': self.data, 'mappings': self.labels}, f, indent=4)

    def train(self):
        if self.data and self.labels:
            self.model.fit(self.data, self.labels)

    def predict(self, field):
        return self.model.predict([field])[0]

    def update(self, field, mapping):
        self.data.append(field)
        self.labels.append(mapping)
        self.train()

# Usage example
if __name__ == "__main__":
    model = FieldMappingModel()
    model.load_data('/Users/samohan/Code/mapping/field_mappings.json')
    model.train()
    print(model.predict('callerName'))
    model.update('callerName', 'requester_name')
    model.save_data('/Users/samohan/Code/mapping/field_mappings.json')
