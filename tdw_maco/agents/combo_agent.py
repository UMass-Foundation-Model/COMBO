import os
import json
import pickle
from openai import AzureOpenAI, OpenAI
import numpy as np
from PIL import Image
from combo.proposer import Proposer
from combo.cwm import CWM
from combo.ranker import Ranker
from combo.intent import IntentTracker
import base64
from utils.utils import get_ego_topdown, get_overlay_ego_topdown

# Function to encode the image
def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class COMBOAgent:
    """
    The High Level Planner of COMBO, consisting of the following components:
        - Action Proposer VLM: a Vision Language Model that takes in the current observation and propose an action
        - Intent Tracker VLM: a Belief Model that takes in the observation history and infer about other agent's actions
        - Compositional World Model: a Video Model that takes in the current observation and the joint action, simulates the world dynamics
        - Outcome Evaluator VLM: a Ranker that takes in the image outcome and rank it
    """
    def __init__(
        self,
        agent_id, task, logger, output_dir='results',
        max_tokens: int = 512,
        debug_mode: bool = False,
        num_propose: int = 2,
        temperature: float = 0,
        proposer_lm_id: str = "gpt-4",
        belief_lm_id: str = "gpt-4",
        ranker_lm_id: str = "gpt-4",
        lm_source: str = "openai",
        only_propose: bool = False,
        no_belief: bool = False,
        plan_horizon: int = 1,
        plan_beam: int = 1,
        cot: bool = False,
        guidance_weight: int = 10,
        history_horizon: int = 3,
    ):

        self.progress = 0
        self.agents_name = None
        self.num_agents = 2
        self.agent_id = agent_id
        self.agent_type = 'combo_agent'
        self.task = task
        self.recipe = None
        self.logger = logger
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.steps = 0
        self.guidance_weight = guidance_weight
        self.history_horizon = history_horizon

        self.only_propose = only_propose
        self.no_belief = no_belief
        self.plan_horizon = plan_horizon
        self.plan_beam = plan_beam
        self.num_propose = num_propose

        self.temperature = temperature
        self.lm_source = lm_source
        self.output_dir = None
        self.action_history = []
        self.object_in_hand = None
        # self.obs_history = []
        self.top_down_history = None
        self.proposer = Proposer(
            task=task,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
            num_propose=num_propose,
            temperature=temperature,
            lm_id=proposer_lm_id,
            lm_source=lm_source,
            cot=cot,
        )
        if self.only_propose:
            self.cwm = CWM(
                serve="server" in lm_source,
                device="cuda:0",
                target_size=(128, 128),
                sample_per_seq=8,
                guidance_weight=guidance_weight,
                model_id=None,
            )
            return
        self.cwm = CWM(
            serve="server" in lm_source,
            device="cuda:0",
        	target_size = (128, 128),
        	sample_per_seq=8,
        	guidance_weight=guidance_weight,
        )
        self.ranker = Ranker(
            task=task,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
            temperature=temperature,
            lm_id=ranker_lm_id,
            lm_source=lm_source,
            cot=cot,
        )
        if self.no_belief:
            return
        self.intenttracker = IntentTracker(
            task=task,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
            temperature=temperature,
            lm_id=belief_lm_id,
            lm_source=lm_source,
            cot=cot,
        )

    def reset(self, obs, info, output_dir='results'):
        self.agents_name = info["agents_name"]
        self.num_agents = len(self.agents_name)
        self.output_dir = os.path.join(output_dir, f"{self.agent_id}")
        self.steps = 0
        self.action_history = [{"type": None, "prompt": "wait"}]
        self.object_in_hand = None
        # self.obs_history = [None for _ in range(self.history_horizon)]
        self.top_down_history = [None for _ in range(self.history_horizon)]
        if self.task == 'cook':
            self.recipe = info["recipe"]
            self.progress = 0
        else:
            self.recipe = None
        self.proposer.reset(self.recipe, self.agents_name)
        if not self.only_propose:
            self.ranker.reset(self.recipe, self.agents_name)

        if not self.only_propose and not self.no_belief:
            self.intenttracker.reset(self.recipe, self.agents_name)


    def act(self, obs):
        self.steps += 1
        output_dir = os.path.join(self.output_dir, f"step_{self.steps}")
        os.makedirs(output_dir, exist_ok=True)
        if obs['rejected']:
            self.action_history[-1] = {"type": None, "prompt": "wait"}
        if self.action_history[-1]["prompt"].startswith("pick up"):
            self.object_in_hand = self.action_history[-1]["prompt"].split("pick up the ")[1]
        elif self.action_history[-1]["prompt"].startswith("place"):
            if "plate" in self.action_history[-1]["prompt"]:
                assert self.object_in_hand == self.recipe[self.agent_id][self.progress], f"seems not right! object_in_hand: {self.object_in_hand}, recipe: {self.recipe}, progress: {self.progress}"
                self.progress += 1 # self.recipe[self.agent_id][self.progress] is the next one to be placed on the plate
            self.object_in_hand = None

        # step1: estimate world state

        reconstructed_imgs = []
        for i, ego_history_path in enumerate(obs['ego_histories']):
            cur_top_down = get_ego_topdown(self.task, obs['camera_matrix'][i], ego_history_path, ego_history_path.replace("img", "depth"))
            # Image.fromarray(cur_top_down).save(
            #     os.path.join(output_dir, ego_history_path.split("/")[-1].replace("img", "reconstructed")))
            reconstructed_imgs.append(cur_top_down)
        overlay = get_overlay_ego_topdown(reconstructed_imgs)
        overlay_path = os.path.join(output_dir, "overlay_top_down.png")
        Image.fromarray(overlay).save(overlay_path)
        inpainting_top_down = self.cwm.run([[overlay_path, None, "", output_dir, "inpainting_top_down"]])[0]

        # step2: propose actions
        proposes = self.proposer.run([inpainting_top_down], self.agent_id, output_dir, self.progress)[0]
        if self.only_propose:
            action = {"type": None, "prompt": proposes[0]}
            self.action_history.append(action)
            return action

        scores, scores_dict = self.ranker.run([inpainting_top_down], output_dir, [None])

        # step3: track others' intents
        self.top_down_history.append(inpainting_top_down)
        if self.no_belief:
            others_actions = {str(agent_id): {"prompt": "wait"} for agent_id in range(self.num_agents)}
            others_actions.pop(str(self.agent_id))
        else:
            others_actions = self.intenttracker.run(self.top_down_history[-self.history_horizon:], self.agent_id, output_dir)

        # step4: simulate outcomes
        new_plans = []
        imaginator_inputs = []
        for i, propose in enumerate(proposes):
            joint_actions = {str(self.agent_id): {"type": None, "prompt": propose}, **others_actions}
            joint_actions_prompt = self.convert_actions_prompt(joint_actions, self.agent_id)
            imaginator_inputs.append([inpainting_top_down, joint_actions_prompt, None, output_dir, f"horizon_0_beam_0_propose_{i}"])
            new_plans.append([{"outcome": inpainting_top_down, "score": scores[0], "scores_dict": scores_dict[0]}, {"act": propose, "belief": others_actions.copy(), "joint_actions_prompt": joint_actions_prompt, "outcome": None}])

        outcomes = self.cwm.run(imaginator_inputs)
        for i, outcome in enumerate(outcomes):
                new_plans[i][-1]["outcome"] = outcome

        # tree search procedure
        for h in range(self.plan_horizon - 1):
            plans = new_plans.copy()
            new_plans = []
            real_history_horizon = self.history_horizon - h - 1
            # step5: evaluate outcomes
            scores, scores_dict = self.ranker.run([plan[-1]["outcome"] for plan in plans], output_dir, [plan[-2]["scores_dict"] for plan in plans])
            for b, plan in enumerate(plans):
                plan[-1]["score"] = scores[b]
                plan[-1]["scores_dict"] = scores_dict[b]
            plans = sorted(plans, key=lambda x: x[-1]["score"], reverse=False)
            with open(os.path.join(output_dir, f'plans_{h}.json'), 'w') as f:
                json.dump(plans, f, indent=4)

            if len(plans) > self.plan_beam:
                plans = plans[:self.plan_beam]

            imaginator_inputs = []
            all_new_proposes = self.proposer.run([plan[-1]["outcome"] for plan in plans], self.agent_id, output_dir, self.progress)
            all_others_actions = []
            if not self.no_belief:
                for b, plan in enumerate(plans):
                    if real_history_horizon > 0:
                        # obs_history = self.obs_history[-real_history_horizon:] + [step["outcome"] for step in plan]
                        top_down_history = self.top_down_history[-real_history_horizon:] + [step["outcome"] for step in plan[1:]]
                    else:
                        # obs_history = [step["outcome"] for step in plan[-self.history_horizon:]]
                        top_down_history = [step["outcome"] for step in plan[-self.history_horizon:]]
                    others_actions = self.intenttracker.run(top_down_history, self.agent_id, output_dir)
                    all_others_actions.append(others_actions)

            for b, plan in enumerate(plans):
                new_proposes = all_new_proposes[b]
                if not self.no_belief:
                    others_actions = all_others_actions[b]
                else:
                    others_actions = {str(agent_id): {"prompt": "wait"} for agent_id in range(self.num_agents)}
                    others_actions.pop(str(self.agent_id))
                for i, propose in enumerate(new_proposes):
                    joint_actions = {str(self.agent_id): {"type": None, "prompt": propose}, **others_actions}
                    joint_actions_prompt = self.convert_actions_prompt(joint_actions, self.agent_id)
                    imaginator_inputs.append([plan[-1]["outcome"], joint_actions_prompt, None, output_dir, f"horizon_{h+1}_beam_{b}_propose_{i}"])
                    new_plans.append(plan + [{"act": propose, "belief": others_actions.copy(), "joint_actions_prompt": joint_actions_prompt, "outcome": None}])

            outcomes = self.cwm.run(imaginator_inputs)
            for i, outcome in enumerate(outcomes):
                new_plans[i][-1]["outcome"] = outcome

        scores, scores_dict = self.ranker.run([plan[-1]["outcome"] for plan in new_plans], output_dir, [plan[-2]["scores_dict"] for plan in new_plans])
        for b, plan in enumerate(new_plans):
            plan[-1]["score"] = scores[b]
            plan[-1]["scores_dict"] = scores_dict[b]
        new_plans = sorted(new_plans, key=lambda x: x[-1]["score"], reverse=False)

        with open(os.path.join(output_dir, f'plans_final.json'), 'w') as f:
            json.dump(new_plans[0], f, indent=4)
        action = {"type": None, "prompt": new_plans[0][1]["act"]}

        print(action)
        self.action_history.append(action)
        with open(os.path.join(output_dir, f'plans_{self.plan_horizon-1}.json'), 'w') as f:
            json.dump(new_plans, f, indent=4)

        return action

    def convert_actions_prompt(self, actions, agent_id):
        prompt = []
        for i, agent_name in enumerate(self.agents_name):
            if str(i) in actions:
                if i != agent_id and self.conflict(actions[str(agent_id)]["prompt"], actions[str(i)]["prompt"]):
                    prompt.append(f"{agent_name} wait.")
                else:
                    prompt.append(f"{agent_name} {actions[str(i)]['prompt']}.")
            else:
                prompt.append(f"{agent_name} wait.")
        return prompt

    def conflict(self, action1, action2):
        if action1 == "wait" or action2 == "wait":
            return False
        if "place" in action1:
            object, loc = action1.split(" onto the ")
            object = object.split("the ")[1]
        else:
            object = action1.split("the ")[1]
            loc = None
        if "place" in action2:
            object2, loc2 = action2.split(" onto the ")
            object2 = object2.split("the ")[1]
        else:
            object2 = action2.split("the ")[1]
            loc2 = None
        if object == object2:
            print("conflict, same object")
            return True
        if loc == "cutting board" and loc2 == "cutting board":
            print("conflict, same location cutting board")
            return True
        return False
