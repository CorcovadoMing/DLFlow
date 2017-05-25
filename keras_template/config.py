from preprocessing import preprocessing

config = {
    # Functions
    'preprocessing': preprocessing,

    # Parameters
    'save_model': 'result/weight.hdf5',
    'length': 20,
    'batch_size': 4,
    'epochs': 50,
    'samples': 10000,
    'dim': 4
}
