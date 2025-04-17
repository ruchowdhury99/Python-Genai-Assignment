import os
from datetime import datetime

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")
    else:
        print(f"Already exists: {path}")

def scaffold_project(base_output="output"):
    timestamp = datetime.now().strftime("project_%Y%m%d_%H%M")
    project_path = os.path.join(base_output, timestamp)
   
    # Directory structure
    folders = [
        "app",
        "app/models",
        "app/schemas",
        "app/services",
        "app/routers",
        "app/core"
    ]
   
    for folder in folders:
        create_folder(os.path.join(project_path, folder))
        with open(os.path.join(project_path, folder, "__init__.py"), "w") as f:
            f.write("# Init\n")

    # Base files
    files = {
        "app/main.py": '''from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Project scaffolded successfully."}
''',
        ".env": "# Add your environment variables here",
        "requirements.txt": "fastapi\nuvicorn\nsqlalchemy\npython-dotenv\npsycopg2-binary\npydantic\n",
        "README.md": f"# Auto-generated FastAPI Project - {timestamp}\n",
        "run.sh": "uvicorn app.main:app --reload --port 8000"
    }

    for file_path, content in files.items():
        full_path = os.path.join(project_path, file_path)
        with open(full_path, "w") as f:
            f.write(content)
        print(f"Created: {full_path}")
   
    return project_path

# Example standalone test
if __name__ == "__main__":
    path = scaffold_project()
    print(f"✅ Project scaffolded at: {path}")
