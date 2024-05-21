import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import socket
import os
import pickle


def main():
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if os.path.exists("tmp/vqa.sock"):
        os.unlink("tmp/vqa.sock")
    server.bind("tmp/vqa.sock")
    server.listen(0)
    print('ready for connection!')
    # with open("tmp/ready.txt", 'w') as f:
    #     f.write("ready!")
    while True:
        connection, address = server.accept()
        r = connection.recv(1024).decode()
        # if r == "stop":
        #     break
        with open('tmp/content.pkl', 'rb') as f:
            content = pickle.load(f)
        video_path = content['video_path']
        questions = ['what is the video about?', content['question']]
        answers = []
        print('\n'+video_path)
        for i in range(2):
            video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
            if type(video_tensor) is list:
                tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
            else:
                tensor = video_tensor.to(model.device, dtype=torch.float16)
            
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            roles = conv.roles

            print(f"{roles[1]}: {questions[i]}")
            question = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + questions[i]
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            #print('video & question processing done!')
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            outputs = outputs.replace("</s>", "")
            answers.append(outputs)
        reply = f"Segment description: {answers[0]}\nAnswer to the question: {answers[1]}"
        print(reply)
        with open('tmp/content.pkl', 'wb') as f:
            pickle.dump(reply, f)
        connection.send(b'sent')
        r = connection.recv(1024)
        connection.close()


if __name__ == '__main__':
    main()