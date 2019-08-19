import os
import sys

# add project root path
project_path = os.path.dirname(os.path.abspath(__file__))

# add subpackage paths
subdirs = [dir for dir in os.listdir(project_path) if os.path.isdir(dir)]
for subdir in subdirs:
    if subdir not in sys.path:
        sys.path.append(subdir)
