import json
import os

import numpy as np
import requests
import re
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI, OpenAI
import base64


from envs.game_controller import get_proposal_prompt as GameProposerPrompt
from envs.cook_controller import get_proposal_prompt as CookProposerPrompt
# Function to encode the image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Proposer:
    """
    a Vision Language Model that takes in the current observation and propose an action for a given agent
    """
    SERVER_ADDR = "http://localhost:8081"
    def __init__(
        self,
        task: str,
        max_tokens: int = 512,
        debug_mode: bool = False,
        num_propose: int = 2,
        temperature: float = 1,
        top_p: float = 0.9,
        lm_id: str = "gpt-4",
        lm_source: str = "openai",
        cot: bool = False
    ):
        self.agents_name = None
        self.task = task
        if task == "cook":
            self.get_proposer_prompt = CookProposerPrompt
        else:
            self.get_proposer_prompt = GameProposerPrompt
        self.recipe = None
        self.debug_mode = debug_mode
        self.num_propose = num_propose

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
                                "temperature":self.temperature,
                                "top_p":self.top_p,
                                "num_beams":1,
                                "max_new_tokens":self.max_tokens,
                                "use_cache":True,
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
            from llava.constants import (
                IMAGE_TOKEN_INDEX,
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
                IMAGE_PLACEHOLDER,
            )
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

                image_files = img.split(',')
                images = []
                for image_file in image_files:
                    if image_file.startswith("http") or image_file.startswith("https"):
                        response = requests.get(image_file)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(image_file).convert("RGB")
                    images.append(image)
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

    def run(self, img_path, agent_id, save_path, progress):
        prompt = f"{IMAGE_PLACEHOLDER}\n{self.get_proposer_prompt(agent_id, self.recipe, self.agents_name)}"
        prompt = [prompt for _ in range(len(img_path) if type(img_path) is list else 1)]
        if self.debug_mode:
            prompt = input(f"Enter prompt for {img_path}:\n")
            self.generator(prompt, img_path)

        # breakpoint()
        all_responses = self.generator(prompt, img_path) # b,
        # all_responses = [all_responses[b:b + 1] for b in range(len(img_path) if type(img_path) is list else 1)]
        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
            f.write(json.dumps({"prompt": prompt, "img": img_path, "response": all_responses}, indent=4))
        all_proposes = []
        for responses in all_responses:
            initial_proposes = []
            possible_actions = set()
            proposes = []
            possible_private_region_actions = set()
            try:
                # propose = responses.split("\n")[-1]
                obs_text, possible_actions_text = responses.split(" My possible actions are: ")
                possible_actions_text, propose = possible_actions_text.split(". I choose to ")
                possible_actions_text = possible_actions_text.split(", ")
                # split all private region actions into another set
                for action in possible_actions_text:
                    if "the private region" in action:
                        possible_private_region_actions.add(action)
                    elif "wait" in action:
                        continue
                    else:
                        possible_actions.add(action)

                if propose.endswith('.'):
                    propose = propose[:-1]
                initial_proposes.append(propose.lower())
            except Exception as e:
                print(f"Proposer Error: {e}, response {responses}, default to wait")

            while len(initial_proposes) < self.num_propose:
                initial_proposes.append(None)
            for propose in initial_proposes:
                if propose not in proposes and (propose in possible_actions or propose in possible_private_region_actions or propose == "wait"):
                    proposes.append(propose)
                    if propose in possible_actions:
                        possible_actions.remove(propose)
                    elif propose in possible_private_region_actions:
                        possible_private_region_actions.remove(propose)
                else:
                    if initial_proposes.index(propose) == 0:
                        print(f"Propose {propose} not in possible actions {possible_actions}, randomly choose one")
                    if len(possible_actions) > 0:
                        propose = np.random.choice(list(possible_actions))
                        # if wait is in the possible actions, drop it
                        # if propose == "wait" and len(possible_actions) > 1:
                        #     propose = np.random.choice(list(possible_actions - {"wait"}))
                        possible_actions.remove(propose)
                        proposes.append(propose)
                    elif len(possible_private_region_actions) > 0:
                        propose = np.random.choice(list(possible_private_region_actions))
                        possible_private_region_actions.remove(propose)
                        proposes.append(propose)
            if len(proposes) < self.num_propose:
                print(f"Proposes: {proposes}, less than {self.num_propose}, append wait")
                if "wait" not in proposes:
                    proposes.append("wait")
            if self.recipe is not None and progress < len(self.recipe[agent_id]) and f"pick up the {self.recipe[agent_id][progress]}" in possible_actions:
                print(f"Proposes: {proposes}, append pick up the {self.recipe[agent_id][progress]}")
                proposes[0] = f"pick up the {self.recipe[agent_id][progress]}"
            # if len(proposes) == 0:
            #     print(f"proposes is empty, default to wait")
            #     proposes = ["wait"]
            print(f"Proposes: {proposes}")
            all_proposes.append(proposes)
        return all_proposes