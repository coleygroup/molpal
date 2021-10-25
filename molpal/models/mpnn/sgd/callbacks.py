__all__ = ["PrintingCallback"]

from typing import Dict, List

from ray.util.sgd.v2 import SGDCallback

class PrintingCallback(SGDCallback):
    def handle_result(self, results: List[Dict], **info):
        print(results)
