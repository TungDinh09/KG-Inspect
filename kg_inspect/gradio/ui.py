# kg_inspect/gradio/ui.py
import os
import gradio as gr

from kg_inspect.gradio.services.pipeline_service import run_pipeline, clear_rag_cache


def _flatten_gradio_message_content(content) -> str:
    """
    Gradio ChatInterface(type='messages') có thể gửi content là str hoặc list block.
    Ta chỉ giữ phần text (bỏ block ảnh).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [b for b in content if isinstance(b, str)]
        return " ".join(text_parts).strip()
    return str(content).strip()


def gradio_history_to_messages(history: list[dict]) -> list[dict]:
    """
    Chuẩn hoá lịch sử từ Gradio (type='messages') thành [{'role','content'}] (text-only).
    """
    msgs: list[dict] = []
    for turn in (history or []):
        role = "user" if turn.get("role") == "user" else "assistant"
        content = _flatten_gradio_message_content(turn.get("content"))
        msgs.append({"role": role, "content": content})
    return msgs


def limit_conversation_history(
    history: list[dict],
    turns: int = 5,
    keep_system: bool = True,
) -> list[dict]:
    """
    turns=5 -> lấy 5 lượt chat gần nhất (user+assistant) ~ 10 messages.
    keep_system=True -> giữ các message role='system' ở đầu (nếu có).
    """
    if not history:
        return []

    system_msgs: list[dict] = []
    rest = history

    if keep_system:
        system_msgs = [m for m in history if m.get("role") == "system"]
        rest = [m for m in history if m.get("role") != "system"]

    max_msgs = max(0, int(turns)) * 2
    if max_msgs <= 0:
        trimmed = []
    else:
        trimmed = rest[-max_msgs:] if len(rest) > max_msgs else rest

    return system_msgs + trimmed


# --- UI Definition ---
def create_demo(turns: int | None = None) -> gr.Blocks:
    if turns is None:
        turns = int(os.getenv("HISTORY_TURNS", "5"))

    async def chat_responder(message: dict, history: list, mode: str) -> str:
        user_text = (message.get("text") or "").strip()
        image_paths: list[str] = message.get("files") or []

        if not user_text:
            return "⚠️ Vui lòng nhập câu hỏi."

        msgs = gradio_history_to_messages(history)
        msgs = limit_conversation_history(msgs, turns=turns)

        # --- Debug console ---
        print("--- [Gửi đến Pipeline] ---")
        print(f"Image Paths: {image_paths}")
        print(f"History (trimmed): {msgs}")
        print(f"Mode: {mode}")
        print("--------------------------")

        status_str, answer_str, debug_str = await run_pipeline(
            image_paths=image_paths,
            user_query=user_text,
            history=msgs,
            mode=mode,
            enable_lightrag=True,
        )

        print("--- [Pipeline Status] ---")
        print(status_str)
        print("--- [Pipeline Debug] ---")
        print(debug_str)
        print("-------------------------")

        if not answer_str and "Lỗi" in status_str:
            return status_str

        return answer_str

    async def on_clear_cache() -> str:
        status, debug = await clear_rag_cache()

        if status.startswith("✅"):
            return f"**{status}**"
        return f"**{status}**\n\n```text\n{debug}\n```"

    with gr.Blocks() as demo:
        gr.Markdown("# Giao diện Chat Đa phương thức với Knowledge Graph")
        gr.Markdown("Bạn có thể tùy chọn tải lên **một** hình ảnh để đặt câu hỏi về nội dung của nó.")

        # ---- Controls row (RAG mode + Cache tools) ----
        with gr.Row():
            with gr.Column(scale=2):
                textbox = gr.MultimodalTextbox(
                    file_count="single",
                    file_types=["image"],
                    label="Nhập câu hỏi và tùy chọn tải lên một hình ảnh",
                )

            with gr.Column(scale=1):
                option_dropdown = gr.Dropdown(
                    choices=["hybrid", "local", "global", "mix", "naive", "bypass"],
                    label="Chế độ RAG",
                    value="hybrid",
                    visible=True,
                )

                gr.Markdown("### Cache tools")
               
                clear_btn = gr.Button("🧹 Clear cache", variant="secondary")
                cache_status = gr.Markdown("")

                clear_btn.click(
                    fn=on_clear_cache,
                    inputs=[],
                    outputs=[cache_status],
                )

        # ---- Chat interface ----
        gr.ChatInterface(
            fn=chat_responder,
            type="messages",
            multimodal=True,
            textbox=textbox,
            save_history=True,
            additional_inputs=[option_dropdown],
            fill_height=True,
        )

    return demo
