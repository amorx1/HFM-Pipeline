from utilities import parse_args, extract_data
import os
from scipy.io import savemat

def main():
    try:
        os.chdir("DATA")
    except:
        print("INVALID DIRECTORY")

    args = parse_args()
    data = extract_data(args)
    
    savemat("output.mat", data)

if __name__ == "__main__":
    main()