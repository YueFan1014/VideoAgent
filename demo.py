import gradio as gr
import openai
from main import preprocess, ReActAgent
from multiprocessing import Process
import os
import socket
from omegaconf import OmegaConf

config = OmegaConf.load('config/default.yaml')
openai_api_key = config['openai_api_key']
use_reid = config['use_reid']
vqa_tool = config['vqa_tool']
base_dir = config['base_dir']


def ask_question(video_file, question):
    preprocess(video_path_list=[video_file], 
               base_dir=base_dir, 
               show_tracking=False)
    answer, log = ReActAgent(video_path=video_file, question=question, base_dir=base_dir, vqa_tool=vqa_tool, use_reid=use_reid, openai_api_key=openai_api_key)
    base_name = os.path.basename(video_file).replace(".mp4", "")
    reid_file = os.path.join("preprocess", base_name, "reid.mp4")
    return answer, reid_file, log


with gr.Row():
    # Define inputs
    with gr.Column(scale=6):
        video_input = gr.Video(label="Upload a video")
        question_input = gr.Textbox(label="Ask a question")


    # Define output    
    with gr.Column(scale=6):
        output_text = gr.Textbox(label="Answer")
        output_reid = gr.Video(label="Video replay with object re-identifcation")
        output_log = gr.Textbox(label="Inference log")


# Create Gradio interface
gr.Interface(
    fn=ask_question,
    inputs=[video_input, question_input],
    outputs=[output_text, output_reid, output_log],
    title="VideoAgent",
    examples = [
            [f"sample_videos/boats.mp4", "How many boats are there in the video?"],
            [f"sample_videos/talking.mp4",
                "From what clue do you know that the woman with black spectacles at the start of the video is married?"],
            [f"sample_videos/books.mp4",
                "Based on the actions observed, what could be a possible motivation or goal for what c is doing in the video?"],
            [f"sample_videos/painting.mp4",
                "What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?"],
            [f"sample_videos/kitchen.mp4",
                "Is there a microwave in the kitchen?"],
        ],
    description="""### This is the demo of [VideoAgent](https://videoagent.github.io/).
    
    Upload a video and ask a question to get an answer from the VideoAgent."""

).launch(share=True)
