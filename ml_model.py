import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class FieldMappingModel:
    def __init__(self):
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())
        self.data = []
        self.labels = []
        self.generate_default_training_data()  # Generate default training data

    def generate_default_training_data(self):
        # Extensive default training data based on real-world SaaS applications
        default_data = [
            "callerName", "callerEmail", "callerPhone", "short_description", "priority",
            "user_id", "user_email", "user_name", "account_id", "account_name",
            "ticket_id", "ticket_status", "ticket_priority", "created_at", "updated_at",
            "department", "group", "assignee", "requester", "description",
            "category", "subcategory", "item", "due_date", "resolution_time",
            "sla_status", "escalation_level", "tags", "comments", "attachments",
            "incident_id", "incident_state", "problem_id", "change_request_id", "approval_status",
            "workflow_id", "workflow_state", "task_id", "task_status", "task_due_date",
            "project_id", "project_name", "milestone", "phase", "resource",
            "customer_id", "customer_name", "customer_email", "customer_phone", "address",
            "billing_address", "shipping_address", "invoice_id", "invoice_status", "payment_status",
            "subscription_id", "subscription_status", "plan_name", "plan_type", "renewal_date",
            "contract_id", "contract_status", "contract_start_date", "contract_end_date", "vendor",
            "partner", "integration_id", "integration_status", "api_key", "webhook_url",
            "domain", "workspace_id", "workspace_name", "team_id", "team_name",
            "role", "permissions", "access_level", "login_time", "logout_time",
            "session_id", "ip_address", "browser", "os", "device_type",
            "value", "assignment_group", "sla_due", "requestedFor", "requesterPhone",
            "due_date", "private", "Location", "incident_state", "business_impact", "personEmail"
        ]
        default_labels = [
            "requester_name", "requester_email", "requester_phone", "description", "priority_level",
            "user_identifier", "email_address", "full_name", "account_identifier", "account_title",
            "ticket_identifier", "status", "priority_level", "creation_date", "modification_date",
            "department_name", "group_name", "assigned_to", "requested_by", "details",
            "main_category", "sub_category", "item_name", "deadline", "time_to_resolve",
            "service_level_agreement", "escalation_stage", "labels", "notes", "file_attachments",
            "incident_identifier", "state", "problem_identifier", "change_request_identifier", "approval_state",
            "workflow_identifier", "workflow_status", "task_identifier", "task_state", "task_deadline",
            "project_identifier", "project_title", "milestone_name", "project_phase", "resource_name",
            "client_identifier", "client_name", "client_email", "client_phone", "location",
            "billing_location", "shipping_location", "invoice_identifier", "invoice_state", "payment_state",
            "subscription_identifier", "subscription_state", "plan_title", "plan_category", "renewal_date",
            "agreement_identifier", "agreement_state", "agreement_start_date", "agreement_end_date", "supplier",
            "business_partner", "integration_identifier", "integration_state", "api_access_key", "callback_url",
            "domain_name", "workspace_identifier", "workspace_title", "team_identifier", "team_title",
            "user_role", "user_permissions", "access_rights", "login_timestamp", "logout_timestamp",
            "session_identifier", "network_address", "web_browser", "operating_system", "device_category",
            "field_value", "group_assignment", "sla_deadline", "requested_for", "requester_phone_number",
            "deadline", "is_private", "location_name", "incident_status", "impact_on_business", "email_address"
        ]
        self.data.extend(default_data)
        self.labels.extend(default_labels)
        self.train()

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
