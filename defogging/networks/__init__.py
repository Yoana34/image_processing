# networks module initialization
import importlib.util
import os

# 动态导入 MSBDN-RDFF.py
spec = importlib.util.spec_from_file_location(
    "MSBDN_RDFF",
    os.path.join(os.path.dirname(__file__), "MSBDN-RDFF.py")
)
MSBDN_RDFF_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MSBDN_RDFF_module)
MSBDN_RDFF = MSBDN_RDFF_module.Net 