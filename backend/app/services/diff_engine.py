import difflib
from pydantic import BaseModel


class FileChange(BaseModel):
    file_path: str
    action: str
    original_content: str
    new_content: str
    diff_patch: str


class DiffEngine:
    @staticmethod
    def generate_unified_diff(original: str, modified: str, filename: str = "file") -> str:
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}"
        )
        return "".join(diff)

    @classmethod
    def create_file_change(
        cls,
        file_path: str,
        original_content: str,
        new_content: str,
        action: str = "edit"
    ) -> FileChange:
        diff_patch = cls.generate_unified_diff(original_content, new_content, filename=file_path)
        return FileChange(
            file_path=file_path,
            action=action,
            original_content=original_content,
            new_content=new_content,
            diff_patch=diff_patch
        )


diff_engine = DiffEngine()
