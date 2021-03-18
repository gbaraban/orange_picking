import yaml

class ConfigWriter(dict):
    def __init__(self, read_loc, *args, **kwargs):
        super(ConfigWriter, self).__init__(*args, **kwargs)
        self.read_loc = read_loc
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def read_config(self):
        with open(self.read_loc, 'r') as file:
            info = yaml.full_load(file)

            for k, v in info.iteritems():
                print(k, v)

    def write_config(self, write_loc):
        with open(self.read_loc, 'w') as file:
            pass

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]