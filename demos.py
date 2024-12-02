import gradio as gr

# 定义一个函数，根据输入返回不同的组件
def update_row(input_value):
    if input_value == "A":
        return [gr.Textbox(label="Textbox A")]
    elif input_value == "B":
        return [gr.JSON(value={"key": "value"})]

with gr.Blocks() as demo:
    dropdown = gr.Dropdown(["A", "B"], label="Select an option")
    with gr.Row() as row:
        pass  # 创建一个空的行

    # 在这里使用change方法，因为它在gr.Blocks上下文中
    dropdown.change(fn=update_row, inputs=dropdown, outputs=row)

demo.launch()