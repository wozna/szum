class Architecture:
    __full_config_file_name = "full_config.txt"
    __opt_config_file_name = "opt_config.txt"
    __layers_config_file_name = "layers_config.txt"
    __base_model_names = ['densenet', 'densenet121', 'densenet169', 'densenet201', 'inception_resnet_v2', 'inception_v3',
                        'NASNet', 'resnet50', 'vgg16', 'vgg19', 'xception', ]

    def __init__(self, model, path):
        self.__model = model
        self.__full_config = model.get_config()
        self.__layers_list = self.__list_layers()
        self.__opt_config = model.optimizer.get_config()
        self.__path = path

    def log(self):
        with open(self.__path + self.__layers_config_file_name, 'w') as f:
            for layer in self.__layers_list:
                f.write("%s\n" % layer)

        f = open(self.__path + self.__full_config_file_name, "w")
        f.write(str(self.__full_config))
        f.close()

        f = open(self.__path + self.__opt_config_file_name, "w")
        f.write(str(self.__opt_config))
        f.close()

    def __list_layers(self):
        layers_list = []
        for layer in self.__model.layers:
            if any(layer.name in s for s in self.__base_model_names):
                for l in layer.layers:
                    layers_list.append(l.name + ' - ' + 'trainable: ' + str(l.trainable))
            else:
                layers_list.append(layer.name + ' - ' + 'trainable: ' + str(layer.trainable))
        return layers_list
