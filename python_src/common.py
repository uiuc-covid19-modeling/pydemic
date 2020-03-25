

class AttrDict(dict):
    expected_kwargs = set()

    def __init__(self, *args, **kwargs):
        if self.expected_kwargs.issubset(set(kwargs.keys())):
            raise ValueError

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

