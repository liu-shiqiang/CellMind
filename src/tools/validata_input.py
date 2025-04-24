from pathlib import Path
import json
from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import tool


class ValidateArgs(BaseModel):
    file_path: str = Field(description="Absolute or relative path to a .h5ad file.")


@tool(
    name="validate_input_file",
    description=(
        "Verify that the provided file exists on disk and has a '.h5ad' suffix. "
        "Return a JSON object {validated: bool, file_path: str}."
    ),
    args_schema=ValidateArgs,
)
def validate_input_file(file_path: str) -> str:
    """
    Ensure the file exists and is in .h5ad format.

    Returns
    -------
    str  (JSON) :
        {"validated": true, "file_path": "<absolute path>"}
        If validation fails the function raises ValueError / FileNotFoundError.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"Expected '.h5ad' suffix, got '{path.suffix}'")

    result = {"validated": True, "file_path": str(path)}
    return json.dumps(result)
