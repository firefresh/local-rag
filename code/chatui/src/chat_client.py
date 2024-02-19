"""The API client for the langchain-esque service."""
import logging
import mimetypes
import typing

import requests

_LOGGER = logging.getLogger(__name__)


class ChatClient:
    """A client for connecting the the lanchain-esque service."""

    def __init__(self, server_url: str, model_name: str) -> None:
        """Initialize the client."""
        self.server_url = server_url
        self._model_name = model_name
        self.default_model = "llama2-7B-chat"

    @property
    def model_name(self) -> str:
        """Return the friendly model name."""
        return self._model_name


    def search(
        self, prompt: str
    ) -> typing.List[typing.Dict[str, typing.Union[str, float]]]:
        """Search for relevant documents and return json data."""
        data = {"content": prompt, "num_docs": 4}
        headers = {
            "accept": "application/json", "Content-Type": "application/json"
        }
        url = f"{self.server_url}/documentSearch"
        _LOGGER.debug(
            "looking up documents - %s", str({"server_url": url, "post_data": data})
        )

        try:
            with requests.post(url, headers=headers, json=data, timeout=30) as req:
                    req.raise_for_status()
                    response = req.json()
                    return typing.cast(
                        typing.List[typing.Dict[str, typing.Union[str, float]]], response
                    )
        except Exception as e:
            _LOGGER.error(f"Failed to get response from /documentSearch endpoint of chain-server. Error details: {e}. Refer to chain-server logs for details.")
            return typing.cast(
                typing.List[typing.Dict[str, typing.Union[str, float]]], []
            )


    def predict(
        self, query: str, use_knowledge_base: bool, num_tokens: int, temperature: float
    ) -> typing.Generator[str, None, None]:
        """Make a model prediction."""
        data = {
            "question": query,
            "context": "",
            "use_knowledge_base": use_knowledge_base,
            "num_tokens": num_tokens,
            "temperature": temperature
        }
        url = f"{self.server_url}/generate"
        _LOGGER.debug(
            "making inference request - %s", str({"server_url": url, "post_data": data})
        )

        try:
            _LOGGER.debug("Decoding response...")
            with requests.post(url, stream=True, json=data, timeout=30) as req:
                    req.raise_for_status()
                    for chunk in req.iter_content(16):
                        _LOGGER.debug("Iterating chunk: {chunk}")
                        yield chunk.decode("UTF-8")
        except Exception as e:
            _LOGGER.error(f"Failed to get response from /generate endpoint of chain-server. Error details: {e}. Refer to chain-server logs for details.")
            yield str("Failed to get response from /generate endpoint of chain-server. Check if the fastapi server in chain-server is up. Refer to chain-server logs for details.")

        # Send None to indicate end of response
        yield None



    def upload_documents(self, file_paths: typing.List[str]) -> None:
        """Upload documents to the kb."""
        url = f"{self.server_url}/uploadDocument"
        headers = {
            "accept": "application/json",
        }

        try:
            for fpath in file_paths:
                mime_type, _ = mimetypes.guess_type(fpath)
                # pylint: disable-next=consider-using-with # with pattern is not intuitive here
                files = {"file": (fpath, open(fpath, "rb"), mime_type)}

                _LOGGER.debug(
                    "uploading file - %s",
                    str({"server_url": url, "file": fpath}),
                )

                resp = requests.post(
                    url, headers=headers, files=files, timeout=600  # type: ignore [arg-type]
                )
                if resp.status_code == 500:
                     raise ValueError(f"{resp.json().get('message', 'Failed to upload document')}")
        except Exception as e:
            _LOGGER.error(f"Failed to get response from /uploadDocument endpoint of chain-server. Error details: {e}. Refer to chain-server logs for details.")
            raise ValueError(f"{e}")