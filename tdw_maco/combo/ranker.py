import json
import os

import numpy as np
import requests
import re
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI, OpenAI
import base64
import cv2


from envs.game_controller import get_value_prompt as GameValuePrompt
from envs.cook_controller import get_value_prompt as CookValuePrompt

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
# Function to encode the image
def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Ranker:
    """
    a Vision Language Model that evaluate an image outcome with a numerical score
    """
    SERVER_ADDR = "http://localhost:8082"
    def __init__(
        self,
        task: str,
        max_tokens: int = 512,
        debug_mode: bool = False,
        temperature: float = 0,
        top_p: float = 0.9,
        lm_id: str = "gpt-4",
        lm_source: str = "openai",
        cot: bool = False,
    ):
        self.progress = None
        self.agents_name = None
        self.recipe = None
        self.task = task
        if task == "cook":
            self.get_value_prompt = CookValuePrompt
        else:
            self.get_value_prompt = GameValuePrompt
        self.debug_mode = debug_mode

        self.lm_id = lm_id
        self.lm_source = lm_source
        self.lm_base = None
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = 3
        self.cot = cot

        if "llava" in self.lm_source:
            self.generate_config = {
                "do_sample": True if self.temperature > 0 else False,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_beams": 1,
                "max_new_tokens": self.max_tokens,
                "use_cache": True,
                "num_return_sequences":1,
            }
        if self.lm_source == "openai":
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
                max_retries=self.max_retries,
                )
        elif self.lm_source == "azure":
            pass
        elif self.lm_source == "huggingface":
            # self.client = AutoModelForCausalLM.from_pretrained(self.lm_id)
            pass
        elif self.lm_source == "llava":
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.conversation import conv_templates, SeparatorStyle
            import torch
            from llava.mm_utils import (
                process_images,
                tokenizer_image_token,
                get_model_name_from_path,
                KeywordsStoppingCriteria,
            )
            self.model_name = get_model_name_from_path(self.lm_id)
            if 'lora' in self.model_name and '7b' in self.model_name:
                self.lm_base = "liuhaotian/llava-v1.5-7b"
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path=self.lm_id, model_base=self.lm_base, model_name=self.model_name, ) # load_4bit=True)
        elif self.lm_source == "llava_server":
            pass
        else:
            raise NotImplementedError(f"{self.lm_source} is not supported!")


        def lm_engine(source, lm_id):

            def openai_generate(prompt, img):
                response = self.client.chat.completions.create(
                    model=lm_id,
                    messages=[
                        # {"role": "user", "content": ""},
                        {"role": "user",
                         "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"},
                                # "detail": "low"
                            },
                         ],
                         },
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    )
                with open(f"combo/chat_raw.jsonl", 'a') as f:
                    f.write(response.model_dump_json(indent=4))
                    f.write('\n')
                usage = dict(response.usage)
                response = response.choices[0].message.content
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                return response

            

            def llava_generate(prompt, img):
                # print(self.model.config)
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if IMAGE_PLACEHOLDER in prompt:
                    if self.model.config.mm_use_im_start_end:
                        prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
                    else:
                        prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
                else:
                    if self.model.config.mm_use_im_start_end:
                        prompt = image_token_se + "\n" + prompt
                    else:
                        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

                if "llama-2" in self.model_name.lower():
                    conv_mode = "llava_llama_2"
                elif "v1" in self.model_name.lower():
                    conv_mode = "llava_v1"
                elif "mpt" in self.model_name.lower():
                    conv_mode = "mpt"
                else:
                    conv_mode = "llava_v0"

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                if type(img) is str:
                    image_files = img.split(',')
                    images = []
                    for image_file in image_files:
                        if image_file.startswith("http") or image_file.startswith("https"):
                            response = requests.get(image_file)
                            image = Image.open(BytesIO(response.content)).convert("RGB")
                        else:
                            image = Image.open(image_file).convert("RGB")
                        images.append(image)
                elif type(img) is np.ndarray:
                    images = [Image.fromarray(img).convert("RGB")]
                else:
                    raise ValueError(f"invalid image type {type(img)}")
                images_tensor = process_images(
                    images,
                    self.image_processor,
                    self.model.config
                ).to(self.model.device, dtype=torch.float16)

                print('======= prompt ======= \n ', prompt)

                input_ids = (
                    tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        images=images_tensor,
                        stopping_criteria=[stopping_criteria],
                        **self.generate_config,
                    )

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(
                        f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                    )
                outputs = self.tokenizer.batch_decode(
                    output_ids[:, input_token_len:], skip_special_tokens=True
                )
                formatted_outputs = []
                conv.remove_last_message()
                for output in outputs:
                    output = output.strip()
                    if output.endswith(stop_str):
                        output = output[: -len(stop_str)]
                    output = output.strip()
                    print('======= response ======= \n ', output)

                    if self.cot:
                        ## extract the action from the response
                        conv_new = conv.copy()
                        conv_new.append_message(conv_new.roles[1], output)
                        conv_new.append_message(conv_new.roles[0], "Answer with only one formatted best action.")
                        conv_new.append_message(conv_new.roles[1], None)
                        prompt = conv_new.get_prompt()

                        print('======= extract prompt ======= \n ', prompt)

                        input_ids = (
                            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                            .unsqueeze(0)
                            .cuda()
                        )

                        stop_str = conv_new.sep if conv_new.sep_style != SeparatorStyle.TWO else conv_new.sep2
                        keywords = [stop_str]
                        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                        with torch.inference_mode():
                            output_ids = self.model.generate(
                                input_ids,
                                images=images_tensor,
                                do_sample=True if self.temperature > 0 else False,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                num_beams=1,
                                max_new_tokens=self.max_tokens,
                                use_cache=True,
                                stopping_criteria=[stopping_criteria],
                            )

                        input_token_len = input_ids.shape[1]
                        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                        if n_diff_input_output > 0:
                            print(
                                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                            )
                        cot_outputs = self.tokenizer.batch_decode(
                            output_ids[:, input_token_len:], skip_special_tokens=True
                        )[0]
                        cot_outputs = cot_outputs.strip()
                        if cot_outputs.endswith(stop_str):
                            cot_outputs = cot_outputs[: -len(stop_str)]
                        cot_outputs = cot_outputs.strip()
                        print('======= formatted response ======= \n ', cot_outputs)
                        formatted_outputs.append(cot_outputs)
                    else:
                        formatted_outputs.append(output)
                with open(f"combo/chat_raw.jsonl", 'a') as f:
                    f.write(json.dumps({"prompt": prompt, "imgs": img, "response": formatted_outputs}, indent=4))
                    f.write('\n')
                return formatted_outputs

            def default_generate(prompt, img):
                return ["wait"]
            def _generate(prompt, img):
                if source == 'openai':
                    return openai_generate(prompt, img)
                elif source == 'azure':
                    raise ValueError("azure is not supported!")
                elif source == 'huggingface':
                    return default_generate(prompt, img)
                    # raise ValueError("huggingface is not supported!")
                elif source == 'llava':
                    return llava_generate(prompt, img)
                elif source == 'llava_server':
                    raise ValueError("server is not supported!")
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.lm_source, self.lm_id)

    def reset(self, recipe, agents_name):
        self.recipe = recipe
        self.agents_name = agents_name
        self.progress = [0 for _ in range(len(agents_name))]

    def run(self, img_path, save_path, previous_scores_dicts):
        # input: img_path
        # output: score [number of the objects to operate, e.g. 5]
        prompt = f"{IMAGE_PLACEHOLDER}\n" + self.get_value_prompt(0, self.recipe, self.agents_name)
        prompt = [prompt for _ in range(len(img_path) if type(img_path) is list else 1)]
        responses = self.generator(prompt, img_path)
        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
            f.write(json.dumps({"prompt": prompt, "img": img_path, "response": responses}, indent=4))
        scores = []
        scores_dict = []
        for i, response in enumerate(responses):
            score_dict = {}
            items = [[],[]]
            score_total = 0
            for line in response.split('\n'):
                item = None
                try:
                    item = line.split(':')[0]
                    score = int(re.search(r'needs (\d+) more step', line).group(1))
                    if previous_scores_dicts[i] is not None and abs(score - previous_scores_dicts[i][item]) > 1:
                        print(f"Ranker score {score} from {line} {img_path[i]} is abnormal compared to previous {previous_scores_dicts[i][item]}, use default")
                        score = previous_scores_dicts[i][item]
                    score_dict[item] = score
                    score_total += score
                    if self.task == "cook":
                        if score == 0:
                            if item in self.recipe[0]:
                                self.progress[0] = max(self.progress[0], self.recipe[0].index(item) + 1)
                            elif item in self.recipe[1]:
                                self.progress[1] = max(self.progress[1], self.recipe[1].index(item) + 1)
                        elif "on the cutting_board" in line:
                            items[0].append(item)
                            items[1].append(item)
                        elif "in the correct agent's" in line:
                            if item in self.recipe[0]:
                                items[0].append(item)
                            elif item in self.recipe[1]:
                                items[1].append(item)
                        elif "in the other agent's" in line:
                            if item in self.recipe[0]:
                                items[1].append(item)
                            elif item in self.recipe[1]:
                                items[0].append(item)
                except Exception as e:
                    print(f"Ranker score not found from {line}, use default, {e}")
                    if previous_scores_dicts[i] is not None and item and item in previous_scores_dicts[i]:
                        score = previous_scores_dicts[i][item]
                    else:
                        print(f"even item is not found, use 10 as default")
                        score = 10
                    score_total += score
                    # score = previous_scores_dicts[i][item]
                if self.task == "cook":
                    to_put = [None, None]
                    if self.progress[0] < len(self.recipe[0]):
                        to_put[0] = self.recipe[0][self.progress[0]]
                    if self.progress[1] < len(self.recipe[1]):
                        to_put[1] = self.recipe[1][self.progress[1]]
                    if len(items[0]) == 5 and to_put[0] not in items[0]:
                        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
                            f.write(json.dumps({"img": img_path, "stuck cutting_board": True}, indent=4))
                        score_total += 4
                    if len(items[1]) == 5 and to_put[1] not in items[1]:
                        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
                            f.write(json.dumps({"img": img_path, "stuck cutting_board": True}, indent=4))
                        score_total += 4
            scores_dict.append(score_dict)
            scores.append(score_total)
        return scores, scores_dict