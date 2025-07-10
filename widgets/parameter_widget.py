import os
import csv
import yaml
import pprint
import requests
import importlib
from collections import OrderedDict
from io import StringIO

import ipywidgets as widgets
from IPython.display import display, clear_output

class DictToObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, DictToObject(v) if isinstance(v, dict) else v)
    def __getattr__(self, name):
        return None

class ParameterEditor:
    def __init__(self, report_folder="report", parameter_csv_url=None):
        self.report_folder = report_folder
        self.parameter_csv_url = parameter_csv_url or \
            'https://raw.githubusercontent.com/ModelEarth/RealityStream/main/parameters/parameter-paths.csv'

        self.model_import_paths = {
            "RFC": "sklearn.ensemble.RandomForestClassifier",
            "RBF": "sklearn.ensemble.RandomForestClassifier",
            "LR": "sklearn.linear_model.LogisticRegression",
            "LogisticRegression": "sklearn.linear_model.LogisticRegression",
            "SVM": "sklearn.svm.SVC",
            "MLP": "sklearn.neural_network.MLPClassifier",
            "XGBoost": "xgboost.XGBClassifier"
        }

        self.loaded_model_classes = {}
        self.user_edited = False
        self._initialize_files()
        self._load_initial_state()
        self._setup_widgets()
        self._bind_callbacks()

    def _initialize_files(self):
        model_names = list(self.model_import_paths.keys())
        os.makedirs(self.report_folder, exist_ok=True)
        with open(os.path.join(self.report_folder, "model-options.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['model_name'])
            for model in model_names:
                writer.writerow([model])

    def _load_initial_state(self):
        self.parameter_paths = self._load_parameter_paths_csv(self.parameter_csv_url)
        self.default_name = next(iter(self.parameter_paths))
        self.default_url = self.parameter_paths[self.default_name]
        self._load_and_parse_yaml(self.default_url)

        self.model_names = []
        with open(os.path.join(self.report_folder, "model-options.csv"), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.model_names.append(row['model_name'])

    def _load_and_parse_yaml(self, url):
        self.last_url = url
        text = requests.get(url).text
        full_dict = yaml.safe_load(text) or {}

        self.default_models = full_dict.get('models', [])
        if isinstance(self.default_models, str):
            self.default_models = [self.default_models]
        self.default_models_lower = [m.lower() for m in self.default_models]

        full_dict.pop('models', None)
        self.last_remote_dict = yaml.safe_load(text) or {}
        self.last_edited_dict = full_dict.copy()
        self.last_params_text = yaml.safe_dump(full_dict, sort_keys=False)

    def _load_parameter_paths_csv(self, url):
        resp = requests.get(url)
        resp.raise_for_status()
        reader = csv.reader(StringIO(resp.text))
        return {name: link for name, link in reader if len((name, link)) == 2}

    def _setup_widgets(self):
        self.chooseParams_widget = widgets.Dropdown(
            options=list(self.parameter_paths.keys()),
            value=self.default_name,
            description='Params Path'
        )
        self.parametersSource_widget = widgets.Text(
            value=self.default_url,
            description='Params From',
            layout=widgets.Layout(width='1200px')
        )
        self.params_widget = widgets.Textarea(
            value=self.last_params_text,
            description='Params',
            layout=widgets.Layout(width='1200px', height='200px')
        )
        self.model_checkboxes = [
            widgets.Checkbox(
                value=name.lower() in self.default_models_lower,
                description=name
            ) for name in self.model_names
        ]
        self.model_selection_box = widgets.VBox(self.model_checkboxes)
        self.apply_button = widgets.Button(description='Update', button_style='primary')
        self.output = widgets.Output()

    def _bind_callbacks(self):
        self.chooseParams_widget.observe(self._on_path_change, names='value')
        self.params_widget.observe(self._on_params_change, names='value')
        self.apply_button.on_click(self._on_update_clicked)

    def _on_path_change(self, change):
        if change['name'] == 'value' and change['type'] == 'change':
            name = change['new']
            url = self.parameter_paths[name]
            self.parametersSource_widget.value = url
            self._load_and_parse_yaml(url)

            for cb in self.model_checkboxes:
                cb.value = cb.description.lower() in self.default_models_lower

            self.params_widget.value = yaml.safe_dump(self.last_edited_dict, sort_keys=False)
            self.user_edited = False
            self._save_parameters_to_report()

    def _on_params_change(self, _):
        self.user_edited = True

    def _on_update_clicked(self, _):
        with self.output:
            clear_output()
            print("\n")

            current_url = self.parametersSource_widget.value
            current_text = self.params_widget.value

            if self.user_edited:
                try:
                    current_edit = yaml.safe_load(current_text) or {}
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML: {e}")
                    return
                diffs = self._compute_diffs(self.last_edited_dict, current_edit)
                self._pretty_print_diff("YAML edits", diffs)
                self.last_params_text = current_text
                self.last_edited_dict = current_edit
                self.user_edited = False
            else:
                print("YAML unchanged.\n")

            if current_url != self.last_url:
                print(f"\n=== URL changed ===\n{self.last_url!r} → {current_url!r}\n")
                try:
                    new_remote = yaml.safe_load(requests.get(current_url).text) or {}
                except Exception as e:
                    print(f"Error fetching new parameters: {e}")
                    return
                self._pretty_print_diff(
                    "Remote YAML diffs",
                    self._compute_diffs(self.last_remote_dict, new_remote)
                )
                self.last_url = current_url
                self.last_remote_dict = new_remote

            selected_models = [cb.description for cb in self.model_checkboxes if cb.value]
            self.last_edited_dict['models'] = selected_models

            added = set(m.lower() for m in selected_models) - set(self.default_models_lower)
            removed = set(self.default_models_lower) - set(m.lower() for m in selected_models)
            if added or removed:
                print("\n=== Model Selection Changes ===")
                if added: print("Added models:", list(added))
                if removed: print("Removed models:", list(removed))

            self._load_models(selected_models)
            self.param = DictToObject(OrderedDict(self.last_edited_dict))
            self.save_training = getattr(self.param, 'save_training', False)
            print("save_training:", self.save_training)
            self._save_parameters_to_report()

    def _compute_diffs(self, a, b):
        return [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]

    def _pretty_print_diff(self, title, diffs):
        if not diffs: return
        print(f"\n=== {title} ===")
        for key, old, new in diffs:
            print(f"• {key}:\n    Old: {pprint.pformat(old)}\n    New: {pprint.pformat(new)}\n")

    def _load_models(self, selected_models):
        self.loaded_model_classes.clear()
        for name in selected_models:
            if name not in self.model_import_paths:
                print(f"Unknown model: {name}")
                continue
            try:
                module, cls = self.model_import_paths[name].rsplit('.', 1)
                mod = importlib.import_module(module)
                self.loaded_model_classes[name] = getattr(mod, cls)
                print(f"Loaded {name} from {module}")
            except Exception as e:
                print(f"Failed to import {name}: {e}")

    def _save_parameters_to_report(self):
        path = os.path.join(self.report_folder, 'parameters.yaml')
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.last_edited_dict, f, sort_keys=False)
        print(f"Saved parameters to {path}")

    def show_ui(self):
        display(widgets.VBox([
            self.chooseParams_widget,
            self.parametersSource_widget,
            self.params_widget,
            self.model_selection_box,
            self.apply_button,
            self.output
        ]))
        self._on_update_clicked(None)
