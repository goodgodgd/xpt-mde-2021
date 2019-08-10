import os
import sys

# add current path
package_path = os.path.dirname(os.path.abspath(__file__))
if package_path not in sys.path:
    sys.path.append(package_path)

# add project root path
project_path = os.path.dirname(package_path)
if project_path not in sys.path:
    sys.path.append(project_path)

print("syspath", sys.path)
