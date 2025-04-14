class TraceBuffer:
    def __init__(self):
        self.buffer = []

    def store(self, trace, ltr_score):
        self.buffer.append((trace, ltr_score))

    def retrieve(self):
        return self.buffer

    def clear(self):
        self.buffer = []