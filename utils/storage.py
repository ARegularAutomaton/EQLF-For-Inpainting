class storage():
    def trained_model_path(net="unet", problem="inpaint", operator=None, epochs=5000):
        assert operator != None

        if problem == "inpaint":
            if operator.localised == "true":
                return "./trained_models/{}/{}/localised/{}/".format(problem, net, epochs)
            if operator.localised == "false":
                return "./trained_models/{}/{}/distributed/{}/".format(problem, net, epochs)
            
        if problem == "ct":
            return "./trained_models/{}/{}/{}/{}/".format(problem, net, operator.type, epochs)
    
    def dataset_location():
        return "./datasets/image_set/"
    
    def log_path(net="unet", problem="inpaint", operator=None, epochs=5000):
        assert operator != None

        if problem == "inpaint":
            if operator.localised == "true":
                return "./logs/{}/{}/localised/{}/".format(problem, net, epochs)
            if operator.localised == "false":
                return "./logs/{}/{}/distributed/{}/".format(problem, net, epochs)
            
        if problem == "ct":
            return "./logs/{}/{}/{}/{}/".format(problem, net, operator.type, epochs)
    