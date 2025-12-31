from langchain_core.callbacks.base import BaseCallbackHandler

class SSEStreamer(BaseCallbackHandler):
    def __init__(self):
        self.queue = []

    def on_llm_new_token(self, token: str, **kwargs):
        # Called by LangChain each time a token is generated
        self.queue.append(token)

    def get_tokens(self):
        # Grab all accumulated tokens and flush queue
        tokens = "".join(self.queue)
        self.queue = []
        return tokens
