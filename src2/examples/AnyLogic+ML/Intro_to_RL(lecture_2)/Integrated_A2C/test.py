'''
import sys
project_folder = '/Users/xudongwang/Third year/IE526/sim/Integrated_A2C'
venv_folder = '/Users/xudongwang/Third year/IE526/sim/Integrated_A2C/A2Cenv/lib/python3.11/site-packages'
sys.path.insert(0, project_folder)
sys.path.insert(0, venv_folder)

print(sys.executable)
source /Users/xudongwang/Third year/IE526/sim/Integrated_A2C/A2Cenv/bin/activate
import sys
import subprocess
import os

# Define the path to your virtual environment's activate script
virtualenv_activate = '/Users/xudongwang/Third\ year/IE526/sim/Integrated_A2C/A2Cenv/bin/activate'  # Replace with the actual path

# Check if we are already in a virtual environment; if not, activate it
if not hasattr(sys, 'real_prefix'):
    try:
        activate_cmd = f'source {virtualenv_activate}'  # On macOS and Linux
        #activate_cmd = f'call {virtualenv_activate}'  # On Windows
        subprocess.check_call(activate_cmd, shell=True)
    except Exception as e:
        print(f"Error activating the virtual environment: {e}")
        sys.exit(1)

# Now, your script is running in the virtual environment
# You can continue with your code here

print(sys.executable)
print("Running within the virtual environment")

'''

#!/usr/bin/python3
import sys
import site
import os

# Define the path to your virtual environment's site-packages directory
virtualenv_site_packages = '/Users/xudongwang/Third\ year/IE526/sim/Integrated_A2C/A2Cenv/lib/python3.11/site-packages'  # Replace with the actual path

# Check if we are not already in a virtual environment
if not hasattr(sys, 'real_prefix'):
    # Add the virtual environment's site-packages to sys.path
    sys.path.insert(0, virtualenv_site_packages)
    site.addsitedir(virtualenv_site_packages)

# Continue with your script

print(sys.executable)
print("Running within the virtual environment")
