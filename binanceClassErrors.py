class optionDataNotInitiatedError(Exception):
        def __init__(self, arg):
            self.arg = arg
            super().__init__(self.arg)

class optionRequestError(Exception):
    def __init__(self, arg):
        self.arg = arg
        super().__init__(self.arg)

class symbolNotFoundError(Exception):
    def __init__(self, arg):
        self.arg = arg
        super().__init__(self.arg)

class sideNotFoundError(Exception):
    def __init__(self, arg):
        self.arg = arg
        super().__init__(self.arg)

class optionTypeNotFoundError(Exception):
    def __init__(self, arg):
        self.arg = arg
        super().__init__(self.arg)

class underlyingRequestError(Exception):
    def __init__(self, arg):
        self.arg = arg
        super().__init__(self.arg)
