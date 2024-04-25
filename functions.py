#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def fix_imbalances(vector, value=0, window = 0.4):
    """ Function to remove random elements
    of an array centered around a specific
    value """
    top_threshold = value + window
    lower_threshold = value - window    
    index = np.where((vector > lower_threshold) & (vector < top_threshold))[0]
    np.random.shuffle(index)
    index_to_delete = index[:int(0.5*index.shape[0])]
    return index_to_delete


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

