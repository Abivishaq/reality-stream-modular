import os
import requests

class ReportPreparer:
    def __init__(self, report_folder: str = "report"):
        self.report_folder = report_folder
        self.index_file_path = os.path.join(self.report_folder, "index.html")
        self.readme_path = os.path.join(self.report_folder, "README.md")
        self.template_url = (
            "https://raw.githubusercontent.com/ModelEarth/localsite/refs/heads/main/start/template/report.html"
        )

    def setup(self) -> int:
        """
        Prepares the report folder:
        - Creates it if it doesn't exist
        - Downloads the index.html template if needed
        - Adds a README.md
        Returns the number of files in the folder.
        """
        self._create_folder()
        self._download_index_template()
        self._add_readme()
        return len(os.listdir(self.report_folder))

    def _create_folder(self):
        if not os.path.exists(self.report_folder):
            os.makedirs(self.report_folder)
            print(f"Created new directory: {self.report_folder}")

    def _download_index_template(self):
        if not os.path.exists(self.index_file_path):
            try:
                response = requests.get(self.template_url)
                response.raise_for_status()
                with open(self.index_file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"Downloaded index.html template to {self.index_file_path}")
            except Exception as e:
                print(f"Error downloading template: {e}")

    def _add_readme(self):
        if not os.path.exists(self.readme_path):
            readme_content = "# Run Models Report\n\nThis folder contains generated reports from model executions."
            with open(self.readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            print(f"Created README.md in {self.report_folder}")

