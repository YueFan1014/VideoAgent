from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_openai import OpenAI, ChatOpenAI, AzureChatOpenAI
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
import openai
import time
import numpy as np


os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"


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
        #llm = ChatOpenAI(model='gpt-4', temperature=0.0, openai_api_key=openai_api_key)
        llm = AzureChatOpenAI(temperature=0, openai_api_version='2024-02-01', azure_deployment="gpt-4o-2024-05-13", streaming=False)
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
    with open('prompts/multiple_choice_prompt.txt') as f:
        t = f.read()

    prompt.template = t
    llm = AzureChatOpenAI(temperature=0, openai_api_version='2024-02-01', azure_deployment="gpt-4o-2024-05-13", streaming=False)
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
    question_num = len(video_question_list)
    tot = 0
    for i in range(question_num):
        print(f"video: {video_path_list[i]}")
        print(f"question: {i}")
        answer_path = f"all_datasets/EgoSchema/all_videos/answers/question_{i}_log.txt"
        if os.path.exists(answer_path):
            continue
        try:
            start_time = time.time()
            answer, log = ReActAgent(video_path=video_path_list[i], question=video_question_list[i], base_dir=base_dir, vqa_tool=vqa_tool, use_reid=use_reid, openai_api_key=openai_api_key)
            with open(answer_path, 'w') as f:
                f.write(video_question_list[i])
                f.write(log)
            tot += 1
            end_time = time.time()
            print(f"inference time for question {i}: {np.round(end_time-start_time, 2)} seconds")
            print("total: ", tot)
        except:
            continue
       

if __name__ == '__main__':
    config = OmegaConf.load('config/default.yaml')
    openai_api_key = config['openai_api_key']
    use_reid = config['use_reid']
    vqa_tool = config['vqa_tool']
    base_dir = config['base_dir']
    with open('datasets/EgoSchema/questions.json') as f:
        datasets = json.load(f)
    video_path_list = []
    video_question_list = []
    
    for q in datasets:
        video_id = q["q_uid"]
        video_file = f"datasets/EgoSchema/all_videos/good_clips_git/{video_id}.mp4"
        video_path_list.append(video_file)
        question_content = q["question"]+'\n'
        for i in range(5):
            question_content+=f'{i}: '+q[f"option {i}"]+'\n'
        # question_content is a multiple-choice question as follows:
        # While not explicitly listing the video actions, explain the ultimate objective c tries to achieve throughout the video.
        # 0: The ultimate objective of c is to carefully dust and clean the sewing machine thoroughly.
        # 1: The ultimate objective for c is to skillfully cut and remove the thread from the fabric's weave.
        # 2: C's ultimate objective is to sew two pieces of fabric together.
        # 3: C's ultimate objective is essentially to carefully remove the thread from the fabric's weave.
        # 4: C's ultimate objective is to shift a fabric on the table.
        video_question_list.append(question_content)
   
    print(len(video_path_list))
    print(len(video_question_list))
    
    preprocess(video_path_list=video_path_list, 
            base_dir=base_dir, 
            show_tracking=False)
    
    main(video_path_list=video_path_list, 
         video_question_list=video_question_list,
          base_dir=base_dir, 
          vqa_tool=vqa_tool, 
          use_reid=use_reid, 
          openai_api_key=openai_api_key)