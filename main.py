from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_openai import OpenAI, ChatOpenAI
from tools import ToolKit
import ast
import sys
import os
import json
import re
from io import StringIO
from captioning import Captioning
from segment_feature import SegmentFeature
from tracking import Tracking
from reid import ReID
from multiprocessing import Process
import socket
from omegaconf import OmegaConf



def ReActAgent(video_path, question, base_dir='preprocess', vqa_tool='videollava', use_reid=True, openai_api_key='your_openai_api_key'):
    assert vqa_tool in ['videollava', 'gpt-4v']
    toolkit = ToolKit(video_path=video_path, base_dir=base_dir, vqa_tool=vqa_tool, use_reid=use_reid, openai_api_key=openai_api_key)
    @tool
    def object_memory_querying(question):
        """Given a question about open-vocabulary objects such as 'how many people are there in the video?' or 'In which segments did the brown dog appear?', this tool will give the answer based on the object memory."""
        @tool
        def database_querying(program):
            """given a MySQL program, this tool will query the database and return the results."""
            ans = toolkit.query_database(program=program)
            return '\n'+ans+'\n'
        @tool
        def open_vocabulary_object_retrieval(description):
            """given an open-vocabulary description of an object or a person (frying pan, person in red clothes e.g.), this tool will return the possible candidate object IDs that satisfy the description."""
            ans = toolkit.retrieve_candidate_objects(description=description)
            return '\n'+ans+'\n'
        prompt = hub.pull("hwchase17/react")
        with open('prompts/database_query_prompt.txt') as f:
            t = f.read()
        prompt.template = t
        llm = ChatOpenAI(model='gpt-4', temperature=0.0, openai_api_key=openai_api_key)
        tools = [database_querying, open_vocabulary_object_retrieval]
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        original_stdout = sys.stdout
        output_catcher = StringIO()
        sys.stdout = output_catcher
        agent_executor.invoke({"input": question})
        sys.stdout = original_stdout
        output_catcher.seek(0)
        lines = output_catcher.readlines()
        color_pattern = re.compile(r'\x1B\[[0-9;]*[mK]')
        answer = None
        for line in lines:
            print(line)
            if line.startswith("Final Answer: "):
                line = color_pattern.sub('', line)
                line = line.replace("Final Answer: ", "")
                answer = line
        return answer

    @tool
    def segment_localization(description):
        """Given a textual description, this tool will return the top-5 candidate segments that are most relevant to the description."""
        answer = toolkit.segment_localization(description, k=5)
        return '\n'+answer+'\n'

    @tool
    def caption_retrieval(input_tuple):
        """given an input tuple (start_segment_ID, end_segment_ID), this tool will retrieve all the captions between the two segments, 15 captions at most. end_segment_ID < start_segment_ID + 15."""
        input_tuple = ast.literal_eval(input_tuple)
        if len(input_tuple) != 2:
            return "\nInvalid input tuple!\n"
        answer = toolkit.caption_retrieval(int(input_tuple[0]), int(input_tuple[1]))
        return '\n'+answer+'\n'

    @tool
    def visual_question_answering(input_tuple):
        """Given an input tuple (question, segment_ID), this tool will focus on the video segments starting from segment_ID-1 to segment_ID+1. It will return the description of the video segment and the answer to the question based on the segment."""
        input_tuple = ast.literal_eval(input_tuple)
        if len(input_tuple) != 2:
            return "\nInvalid input tuple!\n"
        question = input_tuple[0]
        segment_id = int(input_tuple[1])
        answer = toolkit.visual_question_answering(question, segment_id)
        return '\n'+answer+'\n'

    prompt = hub.pull("hwchase17/react")
    with open('prompts/prompt.txt') as f:
        t = f.read()

    prompt.template = t
    #print(prompt)
    llm = ChatOpenAI(model='gpt-4', temperature=0.0, openai_api_key=openai_api_key)
    tools = [caption_retrieval, segment_localization, visual_question_answering, object_memory_querying]
    agent = create_react_agent(llm, tools, prompt)
    print(question)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    original_stdout = sys.stdout
    output_catcher = StringIO()
    sys.stdout = output_catcher
    agent_executor.invoke({"input": question})
    sys.stdout = original_stdout

    output_catcher.seek(0)
    lines = output_catcher.readlines()
    color_pattern = re.compile(r'\x1B\[[0-9;]*[mK]')
    answer = None
    log = ""
    for line in lines:
        print(line)
        new_line = color_pattern.sub('', line)
        log += new_line
        if new_line.startswith("Final Answer: "):
            answer = new_line.replace("Final Answer: ", "")
    return answer, log


def preprocess(video_path_list, base_dir='preprocess', show_tracking=False):
    def check_has_been_preprocessed(video_path):
        base_name = os.path.basename(video_path).replace(".mp4", "")
        video_dir = os.path.join(base_dir, base_name)
        if not os.path.exists(video_dir):
            return False
        files = os.listdir(video_dir)
        required_files = [
            "captions.json", 
            "segment_textual_embedding.pkl", 
            "segment_visual_embedding.pkl", 
            "segment2id.json",
            "tracking.pkl",
            "reid.pkl",
            "tid2clip.pkl",
            "tid2dinov2.pkl",
            "uid2clip.pkl",
            "reid.mp4"
            ]
        for f in required_files:
            if f not in files:
                return False
        return True

    preprocess_list = []
    for video_path in video_path_list:
        if not check_has_been_preprocessed(video_path):
            preprocess_list.append(video_path)

    # build temporal memory
    if len(preprocess_list) == 0:
        return
    captioning = Captioning(video_path_list=preprocess_list, 
                            base_dir=base_dir)
    captioning.run()
    temporal_feature = SegmentFeature(video_path_list=preprocess_list, 
                                      base_dir=base_dir)
    temporal_feature.run()
    # build object memory
    tracking = Tracking(video_path_list=preprocess_list, 
                        base_dir=base_dir,
                        tracking_fps=15, 
                        sample_num=5, 
                        show=show_tracking)
    tracking.run()
    reid = ReID(video_path_list=preprocess_list, 
                base_dir=base_dir)
    reid.run()
    


def main(video_path_list, video_question_list, base_dir='preprocess', vqa_tool='videollava', use_reid=True, openai_api_key='your_openai_api_key'):
    preprocess(video_path_list=video_path_list, 
               base_dir=base_dir, 
               show_tracking=False)
    question_num = len(video_question_list)
    for i in range(question_num):
       ReActAgent(video_path=video_path_list[i], question=video_question_list[i], base_dir=base_dir, vqa_tool=vqa_tool, use_reid=use_reid, openai_api_key=openai_api_key)


if __name__ == '__main__':
    config = OmegaConf.load('config/default.yaml')
    openai_api_key = config['openai_api_key']
    use_reid = config['use_reid']
    vqa_tool = config['vqa_tool']
    base_dir = config['base_dir']

    video_path_list = [
        "sample_videos/boats.mp4",
        "sample_videos/talking.mp4",
        "sample_videos/books.mp4",
        "sample_videos/painting.mp4",
        "sample_videos/kitchen.mp4"
    ]
    video_question_list = [
        "How many boats are there in the video?",
        "From what clue do you know that the woman with black spectacles at the start of the video is married?",
        "Based on the actions observed, what could be a possible motivation or goal for what c is doing in the video?",
        "What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?",
        "Is there a microwave in the kitchen?"
    ]
    main(video_path_list=video_path_list, 
         video_question_list=video_question_list,
          base_dir=base_dir, 
          vqa_tool=vqa_tool, 
          use_reid=use_reid, 
          openai_api_key=openai_api_key)