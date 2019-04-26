import tflearn

def modelfully1(input):
    # Building the encoder
    encoder = tflearn.input_data(shape=[None, input])
    encoder = tflearn.fully_connected(encoder, 400)
    encoder = tflearn.fully_connected(encoder, 200)
    # Building the decoder
    decoder = tflearn.fully_connected(encoder, 400)
    decoder = tflearn.fully_connected(decoder, input, activation='sigmoid')

    # Regression, with mean square error
    net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                             loss='mean_square', metric=None)
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model, encoder, decoder
