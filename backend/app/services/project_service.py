import os
import shutil
from typing import List, Dict, Any
from backend.app.config import settings


class ProjectService:
    def __init__(self, root_dir: str = settings.PROJECTS_DIR):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def _get_project_dir(self, project_id: str) -> str:
        path = os.path.join(self.root_dir, project_id)
        os.makedirs(path, exist_ok=True)
        return path

    def create_project(self, project_id: str, name: str) -> str:
        proj_dir = self._get_project_dir(project_id)
        readme_path = os.path.join(proj_dir, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"# {name}\n\nProject created with CodeMind AI.\n")
        return proj_dir

    def write_file(self, project_id: str, relative_path: str, content: str) -> str:
        proj_dir = self._get_project_dir(project_id)
        full_path = os.path.normpath(os.path.join(proj_dir, relative_path))

        if not full_path.startswith(proj_dir):
            raise ValueError("Path traversal attempt blocked.")

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return relative_path

    def read_file(self, project_id: str, relative_path: str) -> str:
        proj_dir = self._get_project_dir(project_id)
        full_path = os.path.normpath(os.path.join(proj_dir, relative_path))

        if not full_path.startswith(proj_dir) or not os.path.exists(full_path):
            raise FileNotFoundError(f"File '{relative_path}' not found in project.")

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def delete_file(self, project_id: str, relative_path: str) -> bool:
        proj_dir = self._get_project_dir(project_id)
        full_path = os.path.normpath(os.path.join(proj_dir, relative_path))

        if not full_path.startswith(proj_dir):
            raise ValueError("Path traversal attempt blocked.")

        if os.path.isfile(full_path):
            os.remove(full_path)
            return True
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
            return True
        return False

    def list_files(self, project_id: str) -> List[Dict[str, Any]]:
        proj_dir = self._get_project_dir(project_id)
        file_tree = []

        for root, dirs, files in os.walk(proj_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if file.startswith("."):
                    continue
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, proj_dir)
                size = os.path.getsize(full_p)
                file_tree.append({
                    "path": rel_p,
                    "size": size,
                    "is_directory": False
                })

        return file_tree


project_service = ProjectService()
