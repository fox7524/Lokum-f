import os
import re

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
    for f in files:
        if f.endswith((".py", ".md", ".txt", ".json", ".rtf", ".ino")):
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                
                new_content = content
                new_content = new_content.replace("Lokum-F", "Lokum-F")
                new_content = new_content.replace(".lokumf", ".lokumf")
                new_content = new_content.replace("LOKUMF_", "LOKUMF_")
                new_content = new_content.replace("lokumf_", "lokumf_")
                new_content = re.sub(r"\blokumai\b", "lokumf", new_content)
                new_content = re.sub(r"\bLokumai\b", "Lokum-f", new_content)

                if new_content != content:
                    with open(path, "w", encoding="utf-8") as file:
                        file.write(new_content)
                    print(f"Updated {path}")
            except Exception as e:
                pass
