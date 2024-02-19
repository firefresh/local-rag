"""This module contains the chatui gui for having a conversation."""
import functools
import logging
from typing import Any, Dict, List, Tuple, Union

import gradio as gr

from src import assets, chat_client

_LOGGER = logging.getLogger(__name__)
PATH = "/converse"
TITLE = "Converse"
OUTPUT_TOKENS = 1024
MAX_DOCS = 5

_LOCAL_CSS = """

#contextbox {
    overflow-y: scroll !important;
    max-height: 600;
}
"""


def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """Buiild the gradio page to be mounted in the frame."""
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        # create the page header
        gr.Markdown(f"# {TITLE}")
        
        with gr.Tab("Chat"):
            # chat logs
            with gr.Row(equal_height=True):
                chatbot = gr.Chatbot(scale=2, height=600, label=client.model_name)
                latest_response = gr.Textbox(visible=False)
                context = gr.JSON(
                    scale=1,
                    label="Knowledge Base Context",
                    visible=False,
                    elem_id="contextbox",
                )

            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press ENTER",
                    container=False,
                    min_width=600,
                    scale=10,
                    lines=5
                )
                kb_checkbox = gr.Checkbox(
                        label="Use knowledge base", info="", value=False
                )
                submit_btn = gr.Button(value="Submit")

            # user feedback
            with gr.Row():
                # _ = gr.Button(value="👍  Upvote")
                # _ = gr.Button(value="👎  Downvote")
                # _ = gr.Button(value="⚠️  Flag")
                _ = gr.ClearButton(msg)
                _ = gr.ClearButton([msg, chatbot], value="Clear history")
                ctx_show = gr.Button(value="Show Context")
                ctx_hide = gr.Button(value="Hide Context", visible=False)
                kb_go = gr.Button(value="Go to Knowledge Base Management", link="/kb")
        with gr.Tab("Properties"):
            with gr.Column():
                gr.Markdown("Configure chat properties")
                temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, interactive=True, label="Temperature")
                temp_slider.change(lambda x:x, [temp_slider])
                ntokens_slider = gr.Slider(minimum=250, maximum=10000, value=1024, step=250, interactive=True, label="Max tokens")
                ntokens_slider.change(lambda x:x, [ntokens_slider])

        # hide/show context
        def _toggle_context(btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Show Context":
                out = [True, False, True]
            if btn == "Hide Context":
                out = [False, True, False]
            return {
                context: gr.update(visible=out[0]),
                ctx_show: gr.update(visible=out[1]),
                ctx_hide: gr.update(visible=out[2]),
            }

        ctx_show.click(_toggle_context, [ctx_show], [context, ctx_show, ctx_hide])
        ctx_hide.click(_toggle_context, [ctx_hide], [context, ctx_show, ctx_hide])

        # form actions
        _my_build_stream = functools.partial(_stream_predict, client)
        msg.submit(
            _my_build_stream, [kb_checkbox, msg, chatbot, ntokens_slider, temp_slider], [msg, chatbot, context, latest_response]
        )
        submit_btn.click(
            _my_build_stream, [kb_checkbox, msg, chatbot, ntokens_slider, temp_slider], [msg, chatbot, context, latest_response]
        )

    page.queue()
    return page

def _stream_predict(
    client: chat_client.ChatClient,
    use_knowledge_base: bool,
    question: str,
    chat_history: List[Tuple[str, str]],
    num_tokens: int,
    temperature: float 
) -> Any:
    """Make a prediction of the response to the prompt."""
    chunks = ""
    chat_history = chat_history or []
    _LOGGER.info(
        "processing inference request - %s",
        str({"prompt": question, "use_knowledge_base": use_knowledge_base}),
    )

    documents: Union[None, List[Dict[str, Union[str, float]]]] = None
    if use_knowledge_base:
        documents = client.search(prompt = question)

    for chunk in client.predict(
        query=question, 
        use_knowledge_base=use_knowledge_base, 
        num_tokens=int(num_tokens),
        temperature=float(temperature)
    ):
        if chunk:
            chunks += chunk
            yield "", chat_history + [[question, chunks]], documents, ""
        else:
            yield "", chat_history + [[question, chunks]], documents, chunks