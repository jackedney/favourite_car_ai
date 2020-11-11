classes = {
    0: "Porsche 911",
    1: "Honda S2000",
    2: "Neither Porsche 911 nor Honda S2000"
}

def decode_prediction(pred):
    global classes
    # Decodes the prediction of the Porsche vs Honda predictive NeuralNet

    if pred not in list(classes.keys()):
        raise ValueError('`decode_prediction` expects '
                         'a single prediction from 0 to 2'
                         'Found: ' + str(type(pred)))

    result = classes[pred]

    return result