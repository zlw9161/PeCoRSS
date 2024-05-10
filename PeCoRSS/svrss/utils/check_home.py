import pathlib

print(__file__)
SVRSS_HOME = pathlib.Path(__file__).resolve().parents[1]
print(str(SVRSS_HOME))