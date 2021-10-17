# Pass the name of the file to read from as an arguement to 'convert'
def convert(filename):
    f = open(filename, 'r')
    # reading and cleaning the data from the file
    string = f.read().strip()[1:]
    string = string[:-1]
    # Converting into a list of predictions
    preds = string.split(", ")
    # Converting into a 1D Numpy Array
    preds = np.array(preds).astype(int)
    # Returning the array
    return preds
