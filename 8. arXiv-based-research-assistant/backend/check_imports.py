import os
import sys
import importlib

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path}")

# List of modules to check
modules_to_check = [
    "app",
    "app.core.init_app",
    "app.core.processing_queue",
    "app.api.endpoints.chat",
    "app.api.endpoints.papers",
    "app.api.endpoints.concepts",
    "app.db.database",
    "app.models.models",
]

# Check each module
print("\nChecking imports:")
for module_name in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")

# Check if main.py exists
main_path = os.path.join(current_dir, "main.py")
if os.path.exists(main_path):
    print(f"\n✅ main.py exists at {main_path}")
else:
    print(f"\n❌ main.py not found at {main_path}")

# List files in the current directory
print("\nFiles in current directory:")
for file in os.listdir(current_dir):
    print(f"  - {file}")

# List files in the app directory
app_dir = os.path.join(current_dir, "app")
if os.path.exists(app_dir) and os.path.isdir(app_dir):
    print("\nFiles in app directory:")
    for root, dirs, files in os.walk(app_dir):
        rel_path = os.path.relpath(root, current_dir)
        for file in files:
            print(f"  - {os.path.join(rel_path, file)}")
else:
    print("\n❌ app directory not found")
