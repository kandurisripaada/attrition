import os

print("Installing dependencies from requirements.txt...")
os.system("pip install --upgrade pip && pip install -r requirements.txt")
print("Dependencies installed successfully!")
