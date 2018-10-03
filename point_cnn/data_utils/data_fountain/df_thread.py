
import threading


class DfThread(threading.Thread):
    """
    thread to run func, get result after thread finished
    """
    def __init__(self, func, args=()):
        super(DfThread, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(self.args)

    def get_result(self):
        threading.Thread.join(self)     # waiting for thread finished
        try:
            return self.result
        except Exception:
            return None
