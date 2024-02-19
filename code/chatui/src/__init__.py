"""Document Retrieval Service.

Handle document ingestion and retrieval from a VectorDB.
"""

import logging
import os
import sys
import typing

if typing.TYPE_CHECKING:
    from src.api import APIServer


_LOG_FMT = f"[{os.getpid()}] %(asctime)15s [%(levelname)7s] - %(name)s - %(message)s"
_LOG_DATE_FMT = "%b %d %H:%M:%S"
_LOGGER = logging.getLogger(__name__)


def bootstrap_logging(verbosity: int = 0) -> None:
    """Configure Python's logger according to the given verbosity level.

    :param verbosity: The desired verbosity level. Must be one of 0, 1, or 2.
    :type verbosity: typing.Literal[0, 1, 2]
    """
    # determine log level
    verbosity = min(2, max(0, verbosity))  # limit verbosity to 0-2
    log_level = [logging.WARN, logging.INFO, logging.DEBUG][verbosity]

    # configure python's logger
    logging.basicConfig(filename='chatui.log', filemode='w',format=_LOG_FMT, datefmt=_LOG_DATE_FMT, level=log_level)
    # update existing loggers
    _LOGGER.setLevel(logging.DEBUG)
    for logger in [
        __name__,
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
    ]:
        for handler in logging.getLogger(logger).handlers:
            handler.setFormatter(logging.Formatter(fmt=_LOG_FMT, datefmt=_LOG_DATE_FMT))

def main() -> "APIServer":
    """Bootstrap and Execute the application.

    :returns: 0 if the application completed successfully, 1 if an error occurred.
    :rtype: Literal[0,1]
    """
    # boostrap python loggers
    verbosity = int(os.environ.get("APP_VERBOSITY", "1"))
    bootstrap_logging(verbosity)

    # load the application libraries
    # pylint: disable=import-outside-toplevel; this is intentional to allow for the environment to be configured before
    #                                          any of the application libraries are loaded.
    from src import api, chat_client, configuration

    # load config
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    _LOGGER.info("Loading application configuration.")
    config = configuration.AppConfig.from_file(config_file)
    if not config:
        sys.exit(1)
    _LOGGER.info("Configuration: \n%s", config.to_yaml())

    # connect to other services
    client = chat_client.ChatClient(
        f"{config.server_url}:{config.server_port}", config.model_name
    )

    # create api server
    _LOGGER.info("Instantiating the API Server.")
    server = api.APIServer(client)
    server.configure_routes()

    # run until complete
    _LOGGER.info("Starting the API Server.")
    return server