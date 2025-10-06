import os

import gradio as gr

from document_utils import *

# ---------------- Setup ----------------
ensure_upload_folder()
load_embeddings()


# ---------------- Helper Functions ----------------
def refresh_doc_dropdown():
    """Return updated doc list for dropdown"""
    docs = list(document_store.keys())
    if docs:
        return gr.update(choices=docs, value=docs[0])
    return gr.update(choices=[], value=None)


def upload_files(files):
    if not files:
        return gr.update(value="⚠️ No files uploaded."), refresh_doc_dropdown()
    msg, _ = handle_file_upload(files)
    return gr.update(value=f"✅ {msg}"), refresh_doc_dropdown()


def delete_file(filename):
    if not filename:
        return gr.update(value="⚠️ No document selected."), refresh_doc_dropdown()
    msg, _ = delete_document(filename)
    return gr.update(value=f"🟢 {msg}"), refresh_doc_dropdown()


def query_doc(query):
    if not query.strip():
        return "⚠️ Please enter a query."

    fname, chunks = search_documents_topk(query, top_k=3)
    if not chunks:
        return "⚠️ No matching content found."

    combined = " ".join(chunks)
    sentences = combined.split(". ")
    snippet = ". ".join(sentences[:5])  # first 5 sentences

    return snippet + "..."


# ---------------- UI ----------------
def create_ui():
    with gr.Blocks(css="style.css") as app:
        # ---------------- Title ----------------
        gr.Markdown("<h1>📑 Document Q&A System</h1>")

        # ---------------- Login ----------------
        with gr.Group(elem_classes="login-box") as login_col:
            gr.Markdown("### 🔐 Login")
            role_dropdown = gr.Dropdown(choices=["Admin", "User"], label="Select Role")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status", interactive=False)

        # ---------------- Admin Dashboard ----------------
        with gr.Group(visible=False, elem_classes="admin-box") as admin_col:
            gr.Markdown("### 🛠️ Admin Dashboard")
            back_admin_btn = gr.Button("🔙 Back to Login")

            # Upload Section
            with gr.Row():
                file_upload = gr.File(
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".pptx", ".xlsx"],
                )
                upload_btn = gr.Button("Upload Document(s)")
            upload_output = gr.Textbox(
                label="Upload Status", interactive=False, lines=2
            )

            # Delete Section
            doc_list = gr.Dropdown(
                choices=list(document_store.keys()),
                label="Select Document to Delete",
                allow_custom_value=True,
            )
            delete_btn = gr.Button("Delete Document")
            delete_output = gr.Textbox(
                label="Delete Status", interactive=False, lines=2
            )

        # ---------------- User Dashboard ----------------
        with gr.Group(visible=False, elem_classes="user-box") as user_col:
            gr.Markdown("### 🙋 User Dashboard")
            back_user_btn = gr.Button("🔙 Back to Login")
            query_input = gr.Textbox(
                label="Ask a Question", placeholder="Type your query here..."
            )
            query_btn = gr.Button("Search")
            response_output = gr.Textbox(label="Answer", lines=8)

        # ---------------- Login Logic ----------------
        def handle_login(role):
            if role == "Admin":
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "Logged in as Admin",
                )
            elif role == "User":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "Logged in as User",
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "⚠️ Select a role",
                )

        login_btn.click(
            handle_login,
            inputs=[role_dropdown],
            outputs=[login_col, admin_col, user_col, login_msg],
        )

        # ---------------- Back Buttons ----------------
        back_admin_btn.click(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[login_col, admin_col, user_col],
        )
        back_user_btn.click(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[login_col, admin_col, user_col],
        )

        # ---------------- Admin Actions ----------------
        upload_btn.click(
            upload_files, inputs=[file_upload], outputs=[upload_output, doc_list]
        )
        delete_btn.click(
            delete_file, inputs=[doc_list], outputs=[delete_output, doc_list]
        )

        # ---------------- User Actions ----------------
        query_btn.click(query_doc, inputs=[query_input], outputs=[response_output])

        # ---------------- Footer ----------------
        gr.Markdown(
            "<p style='text-align:center; color:#999; margin-top:40px;'>"
            "Powered by Gradio | Dark & Professional UI</p>"
        )

    return app


# ---------------- Main ----------------
if __name__ == "__main__":
    app = create_ui()
    app.launch()
