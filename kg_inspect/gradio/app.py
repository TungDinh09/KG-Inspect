import gradio as gr
from kg_inspect.gradio.ui import create_demo

demo = create_demo()

if __name__ == "__main__":
    demo.launch(show_error=True, debug=True)