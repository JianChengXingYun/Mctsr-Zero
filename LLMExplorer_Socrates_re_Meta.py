##LLMExplorer_Socrates_re_Meta.py

import ast
import asyncio
from collections import defaultdict
import copy
import json
import math
import random
import re
import os
from typing import Dict, List
import uuid
import numpy as np
from pyvis.network import Network
from Swimming_Pool_Async.LLM_Core import LLM_Core
from Swimming_Pool_Async.Prompter import Prompter
from Swimming_Pool_Async.Tools import Tools
from Swimming_Pool_Async.Process_Controller import Process_Controller
import logging
class LLMExplorer_Socrates:
    def __init__(self, llm: LLM_Core, api_llm: LLM_Core = None, initial_threshold=0.3,current_score=None, max_iter=8):
        # 初始化组件
        self.llm = llm
        self.api_llm = api_llm
        self.current_score = current_score
        self.prompter = Prompter()
        self.tools = Tools(filename="",tokenizer = "")
        self.process_controller = Process_Controller(llm = self.llm, tools = self.tools)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_reward = 10
        # 初始化参数
        self.threshold = initial_threshold
        self.max_iter = max_iter
        self.max_expand = 6
        self.class_tag = ""
        self.domain = ""
        self.query = None
        self.iter = 0
        self.use_meta_prompt = True
        self.methods={0: "请在#先前的提示#中添加一个额外的约束/要求：",
                      1: "如果#先前的提示#包含对某些问题的询问，可以增加system提示词的深度和广度。",
                      2: "将一般概念替换为更具体的概念。",
                      3: "如果#先前的提示#可以通过几个简单的思维过程描述，可以重写为明确要求多步骤描述。"}
        # 初始化数据模板
        self.data_template = {
            "model": self.llm.api_model,
            "messages": [],
            "temperature": 0.95,
            "top_p": 0.9,
            "stream": False,
            # "extra_body": {
            # 'repetition_penalty': 1.05,
            # 'top_k': 20,
            # 'min_p': 0.1
            # }
            # "extra_body": {
            #     'stop_token_ids': [self.llm.tokenizer.eos_token_id],
            #     'stop': [self.llm.tokenizer.eos_token],
            # }
        }
        # 初始化数据结构
        self._initialize_data_structures()
        self.Counseling_Report = ""
        self.Meta_Prompt = ""
        self.Next_Meta_Prompt = ""
        self.Meta_reward = 0
    def _initialize_data_structures(self):
        """初始化或重置所有数据结构。"""
        self.to_explore = []
        self.to_explore_reward = {}
        self.history_bank = {}
        self.thinks_bank = {}
        self.ucb_bank = {}
        self.fathers = {}
        self.evaluations_bank = {}
        self.childs = {}
        self.reward_imp_bank = {}
        self.answers_list = []
        self.max_rejected_usage = 1
        self.Meta_Prompt = None
        self.system = None
        self.query = None
        self.visual = False
        self.use_meta_prompt = True

    def reset(self):
        """重置所有储存的数据结构以便重新使用。"""
        print("正在重置 LLMExplorer_Socrates 实例的所有数据结构。")
        self._initialize_data_structures()

    async def get_weak_answer(self, question):
        """获取初始（较弱）答案"""
        print("正在获取弱答案...")
        data_template1 = copy.deepcopy(self.data_template)
        data_template2 = copy.deepcopy(self.data_template)
        if self.use_meta_prompt == True:
            prompt = self.Meta_Prompt
            if self.iter > 0:
                print("树，选择了回到起点，执行元进化...")
                node = self.childs[self.Meta_Prompt][-1]
                Critiques = self.evaluations_bank.get(node, "")
                Critique_C = ""
                for idx, Critique in enumerate(Critiques[-1:]):
                    Critique_C += f"feedback {idx+1}: " + Critique["judge"] + "\n"
                if self.domain == "心理":
                    template = self.prompter.Self_Meta_Refine_Prompt_Psy.format(prompt=self.Meta_Prompt,response=node,Critique=Critique_C)
                
                data_template1["model"] = self.api_llm.api_model
                data_template1["messages"] = [{"role": "user", "content": template}]
                prompt,_ = await self.process_controller.Generate_Response(self.api_llm, data_template1, pattern = r'#+\s*强指令\s*#+(.*)')
                self.Next_Meta_Prompt = prompt.replace("##强指令##", "").replace("<|start|>", "").replace("<|end|>", "")
                print(f"元进化提示词：{self.Next_Meta_Prompt}")
            else:
                if self.domain == "心理":
                    template = self.prompter.Self_Meta_Refine_Prompt_Psy_init.format(prompt=self.Meta_Prompt)
                    pattern = r'#+\s*强指令\s*#+(.*)'
       
                data_template1["model"] = self.api_llm.api_model
                data_template1["messages"] = [{"role": "user", "content": template}]
                prompt,_ = await self.process_controller.Generate_Response(self.api_llm, data_template1, pattern = pattern)
                self.Next_Meta_Prompt = prompt.replace("##强指令##", "").replace("##重写的提示词##", "").replace("<|start|>", "").replace("<|end|>", "")
                print(f"元进化提示词：{self.Next_Meta_Prompt}")
        if self.domain == "心理":
            prompt = self.prompter.sentiment_template.format(Counseling_Report=self.Counseling_Report)
            if self.use_meta_prompt == True:
                data_template2["messages"] = [
                    {"role": "system", "content": self.Next_Meta_Prompt},
                    {"role": "user", "content": prompt}]
            else:
                data_template2["messages"] = [{"role": "user", "content": prompt}]
            result = await self.process_controller.Generate_PsyResponse(self.llm, data_template2)
       
        if "<|_error_|>" not in result:
            print("弱答案获取成功！")
        else:
            print("弱答案获取失败！")
        return result
    
    async def get_enhance_answer(self, weak_answer):
        """基于初始答案和评估生成增强答案。"""
        print("正在生成增强答案...")
        # 获取评估理由
        Critiques = self.evaluations_bank.get(weak_answer, "")
        Critique_C = ""
        for idx, Critique in enumerate(Critiques[-1:]):
            Critique_C += f"feedback {idx+1}: " + Critique["judge"] + "\n"
        data_template2 = copy.deepcopy(self.data_template)
        if self.domain == "心理":
            prompt = self.prompter.Self_Critique_Prompt_Psy.format(multi_dialog=weak_answer,Critique=Critique_C)
            if self.use_meta_prompt == True:
                data_template2["messages"] = [{"role": "system", "content": self.Meta_Prompt},
                                        {"role": "user", "content": prompt}]
            else:
                data_template2["messages"] = [{"role": "user", "content": prompt}]
            result = await self.process_controller.Generate_EnhancePsyResponse(self.llm, data_template2, pattern = r'#+\s*强对话\s*#+(.*)')
       
        if "<|_error_|>" not in result:
            print("增强答案获取成功！")
            return result
        else:
            print("增强答案获取失败！")
            return "<|_error_|>"
 
    async def cal_reward(self, ans):
        print("计算奖励中...")
        if self.domain == "心理":
            data_template3 = copy.deepcopy(self.data_template)
            prompt = self.prompter.Self_Reward_Prompt_Psy.format(multi_dialog=ans)
            data_template3["messages"] = [#{"role": "system", "content": self.system},
                                          {"role": "user", "content": prompt}]
            score, judge = await self.process_controller.Judge_Quantity(self.llm, data_template3)
     
        return score*10, judge

    async def check(self, threshold=None):
        """
        检查是否有任何节点的平均奖励超过整体平均值的指定阈值。

        :param threshold: 阈值，默认为初始阈值。
        :return: 布尔值
        """
        if threshold is None:
            threshold = self.threshold

        if not self.to_explore_reward:
            return False
        
        visit_count = {node: len(self.to_explore_reward[node]) for node in self.to_explore}
        print("每个节点的访问次数:", list(visit_count.values()))
        avg_reward = {
            node: (min(self.to_explore_reward[node]) + np.mean(self.to_explore_reward[node])) / 2#(min(self.to_explore_reward[node]) + np.mean(self.to_explore_reward[node])) / 2
            for node in self.to_explore_reward
        }
        # 提取所有平均奖励值
        avg_reward_values = list(avg_reward.values())

        # 检查是否有足够的数据点来移除最小值
        if len(avg_reward_values) < 2:
            print("平均奖励值的数量不足，无法移除最小值。")
            overall_avg = np.mean(avg_reward_values)
        else:
            # 找到最小值
            min_value = min(avg_reward_values)
            # 移除一个最小值
            avg_reward_values.remove(min_value)
            # 计算移除最小值后的整体平均奖励
            overall_avg = np.mean(avg_reward_values)
            print(f"移除最小值 {min_value} 后的整体平均奖励: {overall_avg}")
        print("每个节点的平均奖励:", list(avg_reward.values()))
        
        # 计算整体平均奖励
        overall_avg = np.mean(list(avg_reward.values()))
        print(f"整体平均奖励: {overall_avg}")
        
        for node, reward in avg_reward.items():
            if reward >= (1 + threshold) * overall_avg:
                print(f"节点 '{node[:30]}' 的奖励 {reward} >=(1 + {threshold}) *  {overall_avg}，返回True")#
                return True
        print(f"没有节点的奖励超过 当前迭代模型的能力{overall_avg}，返回False")#(1 + {threshold}) * {overall_thre}和
        return False
    
    async def update_Meta_nodes(self, old_prompt, new_prompt):
        """
        Recursively update all nodes that reference the old meta prompt
        
        Args:
            old_prompt: The old meta prompt to be replaced
            new_prompt: The new meta prompt that will replace it
        """
        # 1. Update the root node references
        if old_prompt in self.childs:
            self.childs[new_prompt] = self.childs.pop(old_prompt)
        if old_prompt in self.fathers:
            self.fathers[new_prompt] = self.fathers.pop(old_prompt)
        
        # 2. Update all child nodes that reference the old prompt as father
        for node in list(self.fathers.keys()):
            if self.fathers[node] == old_prompt:
                self.fathers[node] = new_prompt
        
        # 3. Update answers_list and to_explore if they contain the old prompt
        if old_prompt in self.answers_list:
            idx = self.answers_list.index(old_prompt)
            self.answers_list[idx] = new_prompt
        
        if old_prompt in self.to_explore:
            idx = self.to_explore.index(old_prompt)
            self.to_explore[idx] = new_prompt
        
        # 4. Update reward and evaluation banks if needed
        if old_prompt in self.to_explore_reward:
            self.to_explore_reward[new_prompt] = self.to_explore_reward.pop(old_prompt)
        
        if old_prompt in self.evaluations_bank:
            self.evaluations_bank[new_prompt] = self.evaluations_bank.pop(old_prompt)
        
        # 5. Update UCB bank if needed
        if old_prompt in self.ucb_bank:
            self.ucb_bank[new_prompt] = self.ucb_bank.pop(old_prompt)

    def draw_tree_pyvis(self, filename="tree.html"):
        """使用pyvis绘制树结构并保存为HTML文件，并标记节点产生的迭代次数和平均分，颜色根据分值变化"""
        net = Network(directed=True, width="100%", height="600px", bgcolor="#222222", font_color="white")

        # 计算每个节点的平均奖励
        # Note: This calculation is slightly different from aggregate_refinement_results
        # as it's specific to the nodes currently in the explorer's banks.
        avg_reward = {}
        for node in self.to_explore_reward:
             if node in self.fathers or node == self.Meta_Prompt: # Only include nodes relevant to the tree
                 rewards_list = self.to_explore_reward[node]
                 if rewards_list:
                    # The original code had: (min(rewards_list) + np.mean(rewards_list)) / 2
                    # Ensure it handles single scores or uses standard average if min isn't appropriate
                    # Let's use the same logic as in aggregate_refinement_results for consistency if possible,
                    # or ensure the logic here is intended for this specific tree visualization.
                    # Sticking to the original logic:
                    if len(rewards_list) > 0:
                         node_min_reward = min(rewards_list)
                         node_avg_reward = np.mean(rewards_list)
                         avg_reward[node] = (node_min_reward + node_avg_reward) / 2
                    else:
                         avg_reward[node] = np.nan # Should not happen if node is in to_explore_reward but defensive

        # Filter out nodes with NaN or no calculated reward for plotting
        plottable_nodes = {node: score for node, score in avg_reward.items() if not np.isnan(score)}

        # Get all plottable scores, used for normalization
        all_scores = list(plottable_nodes.values())
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 1

        # Create a dictionary to record each node's first appearance iteration number
        node_iterations = {}
        for i, answer in enumerate(self.answers_list):
            if answer not in node_iterations:
                node_iterations[answer] = i

        # First add all nodes with iteration info and average score
        for node, score in plottable_nodes.items():
            score_str = f"{score:.2f}"

            # Handle root query node
            if node == self.Meta_Prompt:
                label = "Meta_Prompt (Iter 0)"
                color = "#33FF57"  # Root node fixed green
            else:
                iteration = node_iterations.get(node, "?")

                # Limit label length and add iteration info and average score
                short_text = node[:30] + "..." if len(node) > 30 else node
                label = f"{short_text}\nIter {iteration}, Score: {score_str}"

                # Calculate color based on score (red-yellow-green gradient)
                # Normalize score to [0, 1]
                normalized_score = (score - min_score) / (max_score - min_score) if max_score != min_score else 0.5
                # Calculate RGB color (red -> yellow -> green)
                if normalized_score < 0.5:
                    # Red to yellow (R=255, G=0~255)
                    r = 255
                    g = int(255 * (normalized_score * 2))
                    b = 0
                else:
                    # Yellow to green (R=255~0, G=255)
                    r = int(255 * (1 - (normalized_score - 0.5) * 2))
                    g = 255
                    b = 0
                color = f"#{r:02x}{g:02x}{b:02x}"


            # Add node, also display average score as title (shows on hover)
            title = f"Full text: {node}\nAverage score: {score_str}"
            net.add_node(node, label=label, color=color, title=title)

        # Then add all edges - only for nodes that exist in the network (were added)
        for node in self.fathers.keys():
            father = self.fathers.get(node)
            # Check if both father and node were added to the network before adding edge
            if father and father in net.get_nodes() and node in net.get_nodes():
                net.add_edge(father, node, color="#FF5733")

        # Save HTML file
        net.force_atlas_2based() # Apply layout algorithm
        net.show(filename, notebook=False) # Explicitly set notebook=False when running as script
        print(f"树结构已保存到 {filename}")

    def convert_tuples_to_lists(self, data):
        """递归将字典中的所有元组转换为列表。"""
        if isinstance(data, dict):
            return {k: self.convert_tuples_to_lists(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_tuples_to_lists(item) for item in data]
        elif isinstance(data, tuple):
            return [self.convert_tuples_to_lists(item) for item in data]
        else:
            return data

    async def save_data(self, filename):
        """将数据保存为JSON文件。"""
        final_data = {
            "answers_list": self.answers_list,
            "to_explore": self.to_explore,
            "to_explore_reward": self.to_explore_reward,
            "history_bank": self.history_bank,
            "evaluations_bank": self.evaluations_bank,
            "reward_imp_bank": self.reward_imp_bank,
            "fathers": self.fathers,
            "childs": self.childs,
            "ucb_bank": self.ucb_bank
        }
        # 将元组转换为列表
        final_data_converted = self.convert_tuples_to_lists(final_data)

        # 保存为JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_data_converted, f, ensure_ascii=False, indent=4)
        print(f"最终数据已保存到 {filename}")

    async def rec_data(self):
        """返回转换后的数据字典。"""
        final_data = {
            "answers_list": self.answers_list,
            "to_explore": self.to_explore,
            "to_explore_reward": self.to_explore_reward,
            "history_bank": self.history_bank,
            "evaluations_bank": self.evaluations_bank,
            "reward_imp_bank": self.reward_imp_bank,
            "fathers": self.fathers,
            "childs": self.childs,
            "ucb_bank": self.ucb_bank
        }
        # 将元组转换为列表
        final_data_converted = self.convert_tuples_to_lists(final_data)
        return final_data_converted

    async def get_father_evaluations(self, node):
        """
        遍历给定节点的所有祖先节点，并从 evaluations_bank 收集它们的评估。

        :param node: 当前节点。
        :return: 祖先节点评估的列表。
        """
        temp_evaluations = []
        current_node = node

        while current_node in self.fathers and self.fathers[current_node] is not None:
            parent = self.fathers[current_node]
            parent_evaluations = self.evaluations_bank.get(parent, [])
            temp_evaluations.append(parent_evaluations)
            current_node = parent

        return temp_evaluations[-4:]
    


    def store_history_sync(self, query, answer):
        """记录对话历史（同步方法）。"""
        history = self.history_bank.get(answer, [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        self.history_bank[answer] = history 
        return history

    async def _register_node_and_link_to_parent(self, child, father):
        """
        Helper function to register a new node and link it to its parent.
        Assumes parent_node_key is already in self.fathers or is None (for root).
        """
        self.answers_list.append(child)
        self.to_explore.append(child)
        self.childs[child] = []  # New node is initially a leaf
        self.fathers[child] = father

        if father is not None: # Root node (Meta_Prompt initially) has no parent to add to its child list here
            if father not in self.childs:
                self.childs[father] = []
            self.childs[father].append(child)
        # self.store_history_sync(parent_node_key or self.Meta_Prompt, node_to_add) # History logging can be tricky here

    async def add_to_reward_imp_bank_async(self, weak_answer, reward, answer):
        """记录奖励变化（异步方法）。"""
        if weak_answer not in self.reward_imp_bank:
            self.reward_imp_bank[weak_answer] = []
        self.reward_imp_bank[weak_answer].append((reward, answer))
        return self.reward_imp_bank
        
    def length_and_print(self, node):
        node_length = len(node)
        #print(f"检查节点长度: {node_length}")  # 打印当前节点的长度
        return node_length
    def dialog_length(self, node):
        dialog_list = self.tools.parse_fields(node)
        return len(dialog_list)
    
    async def process_results(self, rho: float = 0) -> List[Dict[str, any]]:#rho: float = 0.08
        try:
            # 计算每个节点的平均奖励np.mean(self.to_explore_reward[node])
            avg_reward = {
                node: (min(self.to_explore_reward[node]) + np.mean(self.to_explore_reward[node])) / 2#(min(self.to_explore_reward[node]) + np.mean(self.to_explore_reward[node])) / 2
                for node in self.to_explore_reward if node != self.Meta_Prompt
            }
            overall_avg = np.mean(list(avg_reward.values()))
            overall_min = min(avg_reward.values())
            print(f"处理时的整体平均奖励: {overall_avg}")
            print(f"整体最小奖励: {overall_min}")
            # 定义长度控制机制的阈值
            Smax = max(avg_reward.values())
            Smin = min(avg_reward.values())
            
            upper_threshold = (1 - rho) * Smax + rho * Smin
            lower_threshold = (1 - rho) * Smin + rho * Smax
            
            print(f"长度控制参数 rho: {rho}")
            print(f"顶层范围阈值 (upper_threshold): {upper_threshold}")
            print(f"被拒绝范围阈值 (lower_threshold): {lower_threshold}")
            
            if upper_threshold <= overall_avg:
                upper_threshold = overall_avg
            if lower_threshold >= overall_avg:
                lower_threshold = overall_avg

            # 筛选符合上层范围的节点作为候选 chosen
            top_candidates = {
                node: reward for node, reward in avg_reward.items()
                if Smax >= reward >= upper_threshold and reward!=0 and reward >= overall_avg
            }

            # 筛选符合下层范围的节点作为候选 rejected
            low_candidates = {
                node: reward for node, reward in avg_reward.items()
                if Smin <= reward <= lower_threshold and reward!=0 and reward < overall_avg
            }

            # 如果没有找到符合范围的候选，则退回到所有节点的最高和最低值
            if not top_candidates:
                top_candidates = avg_reward
            if not low_candidates:
                low_candidates = avg_reward
            if self.domain == "心理":
                highest_node = max(top_candidates.keys(), key=lambda node: self.dialog_length(node))#选择长的
                highest_reward = avg_reward[highest_node]
                lowest_node = min(low_candidates.keys(), key=lambda node: self.dialog_length(node))#选择短的
                lowest_reward = avg_reward[lowest_node] 
            else:
                highest_node = min(top_candidates.keys(), key=lambda node: self.length_and_print(node))#选择长的
                highest_reward = avg_reward[highest_node]
                lowest_node = max(low_candidates.keys(), key=lambda node: self.length_and_print(node))#选择短的
                lowest_reward = avg_reward[lowest_node]
   
            print(f"最高分节点: {highest_node[:30]}, 奖励: {highest_reward}")
            print(f"最低分节点: {lowest_node[:30]}, 奖励: {lowest_reward}")
            if highest_reward == lowest_reward and self.iter!=0:
                print(f"没有奖励区分{highest_reward}, {lowest_reward}")
                return []
            if self.domain == "心理":
                query = self.Counseling_Report
            else:
                query = self.query
           
            pairs = [{
                "prompt": [
                    {"role": "system", "content": self.Meta_Prompt},
                    {"role": "user", "content": query}
                ],
                "chosen": [{"role": "assistant", "content": highest_node}],
                "rejected": [{"role": "assistant", "content": lowest_node}],
                "chosen_reward": highest_reward,
                "rejected_reward": lowest_reward,
                "reward_difference": highest_reward - lowest_reward,
                "domain": self.domain,
                "class_tag": self.class_tag,
                "G_model": str(self.llm.api_model),
                "J_model": str(self.api_llm.api_model),
            }]
            print(f"生成的节点对数: {len(pairs)}")
            return pairs

        except Exception as e:
            print(f"处理结果时发生错误: {e}")
            return []

    async def filter_mature_node(self, max_expand=12):
        """
        过滤尚未达到最大扩展限制的节点。

        :param max_expand: 每个节点的最大子节点数量。
        :return: 过滤后的待探索节点列表。
        """
        filtered_to_explore = [
            node for node in self.to_explore
            if len(self.childs.get(node, [])) < max_expand
        ]
        return filtered_to_explore
    
    
    @staticmethod
    def compute_ucb(r_c, N_n, N_c, C):
        """计算节点的UCB值。"""
        return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))
    
    def get_best_explore_from_ucb(self, to_explore):
        """从待探索节点中选择UCB值最高的节点。"""
        best_node = None
        highest_ucb = float('-inf')
        for node in to_explore:
            ucb_value = self.ucb_bank.get(node, 0)
            if ucb_value > highest_ucb:
                highest_ucb = ucb_value
                best_node = node
        return best_node
        
    async def update_ucb(self, C=2.8):
        """
        更新所有节点的UCB值，包括叶子节点和父节点。

        :param C: 控制探索强度的探索参数。
        """
        # 计算每个待探索节点的访问次数
        visit_count = {node: len(self.to_explore_reward[node]) for node in self.to_explore}
        print("每个节点的访问次数:", list(visit_count.values()))

        # 计算每个节点的平均奖励np.mean(self.to_explore_reward.get(node, [0]))
        avg_reward = {
            node: (min(self.to_explore_reward.get(node, [0])) + np.mean(self.to_explore_reward.get(node, [0]))) / 2#(min(self.to_explore_reward.get(node, [0])) + np.mean(self.to_explore_reward.get(node, [0]))) / 2
            for node in self.to_explore_reward
            }
        print("每个节点的平均奖励:", list(avg_reward.values()))

        # 识别叶子节点（没有子节点的节点）
        leaves = set(self.to_explore) - set(self.fathers.values())
        #print("识别的叶子节点:", leaves)
        # 识别最高奖励的节点
        if avg_reward:
            # 找到具有最高平均奖励的节点
            max_reward = max(avg_reward.values())
            max_nodes = [node for node, reward in avg_reward.items() if reward == max_reward]
            
            # 如果有多个节点具有相同的最高奖励，可以选择打印所有或仅一个
            print(f"提示词：{self.Meta_Prompt[:30]}")
            print("识别的最高奖励的节点:", max_nodes[0][:30])
        else:
            print("没有可用的平均奖励数据来识别最高奖励的节点。")

        # 更新叶子节点的UCB值
        for leaf in leaves:
            father = self.fathers.get(leaf)
            if father is None:
                #N_n = len(self.to_explore)
                N_n = len(self.to_explore)
                N_c = len(self.to_explore_reward.get(leaf, []))
            else:
                N_n = len(self.to_explore_reward.get(father, []))
                N_c = len(self.to_explore_reward.get(leaf, []))
            r_c = avg_reward.get(leaf, 0)
            ucb_value = self.compute_ucb(r_c, N_n, N_c, C)
            self.ucb_bank[leaf] = ucb_value
            print(f"更新叶子节点 '{leaf[:30]}' 的UCB值: {ucb_value}")

        # 递归更新父节点的UCB值
        nodes_to_update = list(leaves)
        processed_nodes = set()  # 添加此集合来记录已处理的节点
        while nodes_to_update:
            new_nodes_to_update = set()
            for node in nodes_to_update:
                father = self.fathers.get(node)
                if father is not None and father not in processed_nodes:
                    # 收集父节点所有子节点的UCB值和奖励
                    child_ucbs = [self.ucb_bank.get(child, 0) for child in self.childs.get(father, [])]
                    child_rewards = [avg_reward.get(child, 0) for child in self.childs.get(father, [])]
                    max_child_reward = max(child_rewards, default=0)
                    father_reward = (avg_reward.get(father, 0) + max_child_reward) / 2
                    #father_reward = avg_reward.get(father, 0)
                    grand_father = self.fathers.get(father, None)
                    # 计算父节点的访问次数
                    #father_count = len(self.to_explore_reward.get(father, []))
                    if grand_father is None: # 'father' is the Meta_Prompt (root)
                        father_N_n = self.iter # Total iterations as "super-root" visits
                    else:
                        father_N_n = visit_count.get(grand_father, 0)
                    father_explore_count = len(self.to_explore_reward.get(father, []))
                    # 计算父节点的UCB值
                    father_ucb_value = self.compute_ucb(father_reward, father_N_n, father_explore_count, C)
                    self.ucb_bank[father] = father_ucb_value
                    print(f"更新父节点 '{father[:30]}' 的UCB值: {father_ucb_value}")

                    # 将父节点添加到已处理集合和下一个更新列表中
                    processed_nodes.add(father)
                    new_nodes_to_update.add(father)
            nodes_to_update = list(new_nodes_to_update)


    async def step(self, weak_answer):
        """执行一次探索步骤以获取增强答案。"""
        if weak_answer == self.Meta_Prompt:
            return await self.get_weak_answer(weak_answer)
        return await self.get_enhance_answer(weak_answer)

    async def sampling_reward(self, answer):
        """评估答案并记录奖励。"""
        if answer not in self.to_explore_reward:
            self.to_explore_reward[answer] = []
        if answer not in self.evaluations_bank:
            self.evaluations_bank[answer] = []
        #if answer == self.Meta_Prompt:
        # 计算每个节点的平均奖励
        # 动态因子 c 的计算：随着迭代次数增加，c 趋近于 1
         # 例如: c = 1.0 + initial_boost / (self.iter + smoothing_constant)
         # iter=0: c = 1.0 + 0.2/1 = 1.2
         # iter=1: c = 1.0 + 0.2/2 = 1.1
         # iter=2: c = 1.0 + 0.2/3 ~ 1.067
         # iter=7: c = 1.0 + 0.2/8 = 1.025
        initial_boost = 0 # 控制初始时的奖励加成幅度
        smoothing_constant = 1 # 避免除以零或过大的初始值，并控制衰减速度
        dynamic_c = 1.0 + initial_boost / (self.iter + smoothing_constant)
        print(f"Iteration {self.iter}, Dynamic factor c: {dynamic_c:.4f}")
        if answer == self.Meta_Prompt:#为什么不每次采样奖励都更新?因为节点访问次数过多导致ucb降分np.mean(self.to_explore_reward.get(node, [0]))
            avg_reward = {
                node: (min(self.to_explore_reward.get(node, [0])) + np.mean(self.to_explore_reward.get(node, [0]))) / 2#(min(self.to_explore_reward.get(node, [0])) + np.mean(self.to_explore_reward.get(node, [0]))) / 2
                for node in self.childs[self.Meta_Prompt]
            }
            child_rewards = np.mean([avg_reward.get(child, 0) for child in self.childs.get(self.Meta_Prompt, [])])# +100 
            #min_rewards = min([avg_reward.get(child, 0) for child in self.childs.get(self.Meta_Prompt, [])])

            print("child_rewards: ", child_rewards)
            #print("min_rewards: ", min_rewards)
            #self.Meta_reward = (min_rewards + child_rewards) / 2
            self.Meta_reward = child_rewards
            print("Meta_rewards: ", self.Meta_reward)
            child_rewards = child_rewards * dynamic_c
            self.to_explore_reward[self.Meta_Prompt].append(child_rewards)
        #return self.to_explore_reward
        ##########放下面是因为如果是query，那么就没有to_explore_reward，所以在计算每个节点的平均奖励时不能计算query的奖励
        else:
            reward, judge = await self.cal_reward(answer)
            self.to_explore_reward[answer].append(reward)
            self.evaluations_bank[answer].append({"reward": reward, "judge": judge})
        return self.to_explore_reward

    async def main_loop(self, inputs, use_weak=True):
        print("max iter:", self.max_iter)
        self.class_tag = inputs.get("class_tag", "通用")
        self.system, query = inputs["prompt"][0]["content"], inputs["prompt"][1]["content"]
        self.domain = inputs.get("domain", "通用")
        
        if self.domain == "心理": # 这部分逻辑可以保留，如果Meta_Prompt的初始值依赖它
            self.Counseling_Report = query
            self.Meta_Prompt = self.system # Initial Meta_Prompt based on system prompt

        # 1. Initial answer generation and Meta Prompt setup
        weak_answer = await self.get_weak_answer(self.Meta_Prompt) if use_weak else inputs["chosen"][-1]["content"]
        if "<|_error_|>" in weak_answer: print("获取初始答案失败"); return "<|_error_|>"

        if self.use_meta_prompt == True:
            evolved_meta_prompt = self.Next_Meta_Prompt # self.Next_Meta_Prompt is set in get_weak_answer
            original_meta_prompt = self.Meta_Prompt
            self.Meta_Prompt = evolved_meta_prompt
            # Register the new Meta_Prompt (if it changed)
            if original_meta_prompt != self.Meta_Prompt:
                # This handles if the text of Meta_Prompt actually changed
                await self.update_Meta_nodes(original_meta_prompt, self.Meta_Prompt)
                self.answers_list.append(self.Meta_Prompt) # Add new meta to lists if it's truly new
                self.to_explore.append(self.Meta_Prompt)
                self.fathers[self.Meta_Prompt] = None # New root
                self.childs[self.Meta_Prompt] = []
            else: # Meta_Prompt text didn't change, ensure it's initialized if first run
                self.answers_list.append(self.Meta_Prompt)
                self.to_explore.append(self.Meta_Prompt)
                self.fathers[self.Meta_Prompt] = None
                self.childs[self.Meta_Prompt] = []


            # Register the initial raw answer as a child of the current self.Meta_Prompt
            await self._register_node_and_link_to_parent(weak_answer, self.Meta_Prompt)
            self.store_history_sync(self.Meta_Prompt, weak_answer)

            await self.sampling_reward(weak_answer)
            await self.sampling_reward(self.Meta_Prompt) # Scores Meta_Prompt based on weak_answer
            await self.update_ucb(C=2.8)
        else:
            self.store_history_sync(query, weak_answer)  # 使用同步方法记录历史
            self.answers_list.append(weak_answer)
            self.to_explore.append(weak_answer)
            self.childs[weak_answer] = []
            self.fathers[weak_answer] = None
            # 评估初始答案
            await self.sampling_reward(weak_answer)
            await self.update_ucb(C=2.8)
        #processed_weak_answers = set()

        for i in range(self.max_iter):
            print(f'迭代 {i + 1}:')
            self.iter = i + 1
            filtered_to_explore = [
                n for n in await self.filter_mature_node(max_expand=self.max_expand)
                #if n not in processed_weak_answers
            ]
            weak_answer = self.get_best_explore_from_ucb(filtered_to_explore)

            if not weak_answer: print("没有可探索的节点，结束循环。"); break

            current_parent_node = weak_answer # This is the node selected for expansion

            # Pre-sample the node to be expanded if it's not the Meta_Prompt
            # (Meta_Prompt's sampling is handled specially below after its child is generated)
            if current_parent_node != self.Meta_Prompt:
                await self.sampling_reward(current_parent_node)

            # --- Core Logic for Expansion or Meta-Evolution ---
            if current_parent_node == self.Meta_Prompt:
                # Meta-evolution step
                meta_performance_before_new_child = self.Meta_reward # From previous Meta_Prompt sampling

                # 1. Generate a new child from the current Meta_Prompt.
                #    self.step calls get_weak_answer, which sets self.Next_Meta_Prompt
                answer = await self.step(current_parent_node) # `answer` is the new child node

                if "<|_error_|>" in answer:
                    print(f"Meta prompt {current_parent_node[:30]} 生成子节点出错。")
                    return "<|_error_|>"
                    #processed_weak_answers.add(current_parent_node); continue
                
                # 2. Score the new child
                await self.sampling_reward(answer)

                # 3. Register the new child and link to current_parent_node (the Meta_Prompt being evolved)
                await self._register_node_and_link_to_parent(answer, current_parent_node)
                self.store_history_sync(current_parent_node, answer)

                # 4. NOW, rescore the current_parent_node (Meta_Prompt).
                #    Its score will now reflect the new child `answer`.
                await self.sampling_reward(current_parent_node)
                # self.Meta_reward is now updated based on all children of current_parent_node

                # 5. Decide if the meta-evolution was successful
                reward_of_new_child = np.mean(self.to_explore_reward.get(answer, [0]))
                if reward_of_new_child >= meta_performance_before_new_child:
                    print("元进化成功，对元提示进行梯度更新...")
                    old_meta_key = current_parent_node
                    new_meta_key = self.Next_Meta_Prompt # From self.step via get_weak_answer
                    
                    if old_meta_key != new_meta_key: # Only if the prompt text actually changed
                        await self.update_Meta_nodes(old_meta_key, new_meta_key)
                        self.Meta_Prompt = new_meta_key # Update the active Meta_Prompt

                        # The new child `answer` now belongs to the `new_meta_key`
                        self.fathers[answer] = self.Meta_Prompt
                        # Ensure linkage (update_Meta_nodes might handle child list transfer, but be explicit)
                        if self.Meta_Prompt not in self.childs: self.childs[self.Meta_Prompt] = []
                        if answer not in self.childs[self.Meta_Prompt]: self.childs[self.Meta_Prompt].append(answer)
                        if old_meta_key in self.childs and answer in self.childs[old_meta_key]:
                            self.childs[old_meta_key].remove(answer)
                # else: Meta_Prompt doesn't change, `answer` remains child of `current_parent_node`
                
            else:
                # Regular node expansion
                answer = await self.step(current_parent_node) # `answer` is the enhanced node

                if "<|_error_|>" in answer:
                    print(f"节点 {current_parent_node[:30]} 扩展时出错。")
                    return "<|_error_|>"
                    #processed_weak_answers.add(current_parent_node); continue

                # Simplified self-reference check (original had more complex loop)
                if answer == current_parent_node:
                    print(f"答案与选定节点相同: {current_parent_node[:30]}，跳过。")
                    # Consider adding current_parent_node to processed_weak_answers or increasing max_expand
                    self.max_expand +=1 # As per your original code for this case

                    #processed_weak_answers.add(current_parent_node); continue 

                await self.sampling_reward(answer)
                await self._register_node_and_link_to_parent(answer, current_parent_node)
                self.store_history_sync(self.Meta_Prompt, answer) # History usually with active Meta

            # --- Common Post-Expansion/Evolution Steps ---
            # `answer` is the newly created node (either from meta-evo or regular expansion)
            # `current_parent_node` is its parent
            
            # Update reward_imp_bank using the actual parent of `answer`
            actual_parent_of_answer = self.fathers.get(answer) # Could be old_meta or new_meta or current_parent_node
            if actual_parent_of_answer and \
               actual_parent_of_answer in self.to_explore_reward and self.to_explore_reward[actual_parent_of_answer] and \
               answer in self.to_explore_reward and self.to_explore_reward[answer]:
                
                reward_difference = min(self.to_explore_reward[answer]) - min(self.to_explore_reward[actual_parent_of_answer])
                await self.add_to_reward_imp_bank_async(actual_parent_of_answer, reward_difference, answer)
            
            #processed_weak_answers.add(current_parent_node) # Mark the expanded node

            # # --- Visualization, UCB Update, Loop Control ---
            if self.visual:
                path = os.path.join(f"tree_{uuid.uuid4()}_{i + 1}.html")
                self.draw_tree_pyvis(path)
            await self.update_ucb(C=2.8)

            if i + 1 > 3 and i + 1 >= self.max_iter:
                return await self.process_results()

        return await self.process_results()

if __name__ == "__main__":
    import asyncio
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    query = {}
    query["domain"] = "心理"
    query["prompt"] = [{"role":"system", "content": "You are a helpful assistant."},
                       {"role":"user", "content": """#### 4.1 Demographic Information: \n张伟，29岁，男性，软件工程师\n#### 4.2 Presenting Problem: \n主诉为无
法控制思绪，感到思维混乱，影响工作和学习。\n#### 4.3 Relevant History: \n张伟自小家庭环境较为紧张，父母对他有较高的期望。他从小成绩优异，但因过度追求完美，逐渐产生焦虑
情绪。在大学期间，曾经历过一次重大的失落，导致情绪低落，未得到及时的心理支持。\n#### 4.4 Current Life Situation: \n目前张伟工作繁忙，常感到时间不够用，难以专注于任务。
由于思绪不清晰，工作表现受到了影响，导致他在团队中的自信心下降，感到孤立无援。"""}]

    # 读取敏感词
    with open('FilterWord.txt', 'r', encoding='utf-8') as f:
        sensitive_words = [line.strip() for line in f.readlines()]

        # # 初始化LLM_Core
    llm = LLM_Core(
        tokenizer,
        use_async=True,
        api_model="gpt-4.1-mini",
        base_url="", ## set yours
        api_key='' ## set yours
    )
    # 初始化LLMExplorer，设置一个合理的max_iter以避免长时间运行
    explorer = LLMExplorer_Socrates(
        llm=llm,
        api_llm=llm,
        max_iter=4  # 设置一个合理的迭代次数
    )
    explorer.visual = True
    # 运行主循环
    dicts = asyncio.run(explorer.main_loop(query))
    print(dicts)