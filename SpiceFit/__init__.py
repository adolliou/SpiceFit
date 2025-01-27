from .core import *
from .util import *
import platform
# def find_xuvtop():
#     xuvtop = os.getenv('XUVTOP')
#
#     if xuvtop:
#         if False:
#             print(f"XUVTOP is already set to: {xuvtop}")
#     elif platform.system() == 'Linux':
#         bashrc_path = os.path.expanduser('~/.bashrc')
#         if os.path.exists(bashrc_path):
#             with open(bashrc_path, 'r') as file:
#                 lines = file.readlines()
#                 for line in lines:
#                     if 'export XUVTOP' in line:
#                         xuvtop_value = line.split('=')[1].strip().strip('"')
#                         os.environ['XUVTOP'] = xuvtop_value
#                         print(f"XUVTOP found in .bashrc and set to: {xuvtop_value}")
#                         return
#         print("XUVTOP not found in environment variables or .bashrc.")
#     else:
#         print("XUVTOP not found and manual search is only supported on Linux.")
#
# # Ensure this function only runs in the main process
# if __name__ == "__main__":
#     find_xuvtop()
