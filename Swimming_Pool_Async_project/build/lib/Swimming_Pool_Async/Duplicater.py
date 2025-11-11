#Duplicater.py
import asyncio
import copy
from cycler import V
import torch
from transformers import AutoTokenizer
from enum import Enum, auto
from tqdm import tqdm
from Swimming_Pool_Async.LLMExplorer import LLMExplorer
from Swimming_Pool_Async.LLMExplorer2 import LLMExplorer2
from Swimming_Pool_Async.LLMExplorer_GPT import LLMExplorer_GPT
from Swimming_Pool_Async.LLMExplorer_Zero import LLMExplorer_Zero
from Swimming_Pool_Async.LLMExplorer_Zero_psy import LLMExplorer_Zero_psy
from Swimming_Pool_Async.LLMExplorer_Socrates import LLMExplorer_Socrates
from Swimming_Pool_Async.LLMExplorer_Socrates_Judge import LLMExplorer_Socrates_Judge
from Swimming_Pool_Async.LLMExplorer_Socrates_Rule import LLMExplorer_Socrates_Rule
from Swimming_Pool_Async.LLMExplorer_Socrates_roleplay import LLMExplorer_Socrates_roleplay
from Swimming_Pool_Async import LLMExplorer_Socrates_re_Meta_flash
from Swimming_Pool_Async.Process_Controller import Process_Controller
from Swimming_Pool_Async.MorphicResonanceKernel import MorphicResonanceKernel
class ProcessStage(Enum):
    # Adding async variants
    REVERSE_REPORT_ASYNC = auto()
    SLOTH_ASYNC = auto()
    REPORT_ASYNC = auto()
    SCENARIO_TO_REPORT_ASYNC = auto()
    SCENARIO_TO_REPORT_IMPEDANCE_ASYNC = auto()
    MEMO_ASYNC = auto()
    DIALOG_ASYNC = auto()
    DIALOG_IMPEDANCE_ASYNC = auto()
    DIALOG_REJECTED_ASYNC = auto()
    BOT_PRE_ASYNC = auto()
    BOT_ASYNC = auto()
    BOT_POST_ASYNC = auto()
    BOT_DIALOG_ASYNC = auto()
    PsyEval_ASYNC = auto()
    PsyEval_IMPEDANCE_ASYNC = auto()
    PsyEvalScore_ASYNC = auto()
    DeleteforEval_ASYNC = auto()
    DeleteforEval_Solution_ASYNC = auto()
    DeleteforEval_TextGrad_ASYNC = auto()
    report_class_ASYNC = auto()
    REWARD_THERAPY_QUALITY_ASYNC = auto()
    REWARD_THERAPY_QUALITY_META_ASYNC = auto()
    REWARD_THERAPY_QUALITY_IMPEDANCE_ASYNC = auto()
    REWARD_THERAPY_REASONING_SYNDATA_ASYNC = auto()
    TEXTGRAD_ASYNC = auto()
    TEXTGRAD_QUALITY_META_ASYNC = auto()
    TEXTGRAD_IMPEDANCE_ASYNC = auto()
    REJECTED_ASYNC = auto()
    CONVERT_TRI_ASYNC = auto()
    CONVERT_REWARD_REASONING_ASYNC = auto()
    CONVERT_GOODORBAD_BY_LABEL_ASYNC = auto()
    DIALOG_REJECTED_COMMON_ASYNC = auto()
    DATA_ENHANCE_ASYNC = auto()
    Scenario_to_Query = auto()
    MemoryFlow = auto()
    ASYNC_SCENARIO_TO_REPORT_IMPEDANCE = auto()
    LLM_EXPLORE_ASYNC = auto()
    PsyEval_IMPEDANCE = auto()
    REWARD_THERAPY_QUALITY = auto()
    CONVERTFOREVAL = auto()
    IDENTITY_GENERATE = auto()
    IDENTITY_REJECTED = auto()
    EVOL_INSTRUCT = auto()
    EVOL_INSTRUCT_AME = auto()
    LLM_EXPLORE_ASYNC2 = auto()
    LLM_EXPLORE_ASYNC_GPT = auto()
    ROLE_GENERATE = auto()
    GenesisRAM = auto()
    Faynman_Technique = auto()
    REWRITE_INSTRUCT = auto()
    QA_REJECTED = auto()
    DIALOG_REJECTED = auto()
    REPORT_CLASS = auto()
    DIALOG_GENERATE = auto()
    Evol_Instruct_noA = auto()
    LLMExplorer_Zero = auto()
    EVOL_MCTSR = auto()
    Intervent = auto()
    LLMExplorer_Zero_psy = auto()
    SOCRATES_INF = auto()
    SOCRATES_INF_RE = auto()
    SOCRATES_FIN = auto()
    META_JUSTICE = auto()
    MATH_TRANSLATION = auto()
    DIALOG_TO_THINKER = auto()
    SOCRATES_INF_ROLE = auto()
    SOCRATES_FIN_ROLE = auto()
    CharacterV1_to_V3 = auto()
    DISTILL_FIN_ROLE = auto()
    TextGrad_FIN = auto()
    TextGrad_INF = auto()
    COUNREPORT_INF = auto()
    DISTILL_INF_ROLE = auto()
    SYNTHETIC_SYSTEM = auto()
    PSY_REJECTED = auto()
    DISTILL_INF = auto()
    MorphicResonanceKernel = auto()
    JUDGE_ANALOGY = auto()
    REWARD_SCORE = auto()
    PsyTextGrad_FIN = auto()
    PsyTextGrad_INF = auto()
    PROMPTREWARD_SCORE = auto()
    SOCRATES_RULE_INF = auto()
    SOCRATES_RULE_FIN = auto()
    TextGrad_Catalyst = auto()
    Rewrite_Dialog = auto()
    Endless_Therapy_Dialog = auto()
    Endless_Therapy_Answer = auto()
    PARALLEL_EVALUATION = auto()
    JUDGE_QUANTITY = auto()
    SOCRATES_FIN_RE = auto()
    SOCRATES_INF_RE_Meta = auto()
    SOCRATES_INF_RE_Meta_THINK = auto()
    SOCRATES_FIN_RE_Meta_THINK = auto()
    SOCRATES_FIN_RE_Meta = auto()
    judge_fabricationDetected = auto()
    judge_chatAbility = auto()
    judge_consultantActiveEndingDetected = auto()
    Rewrite_fabricate = auto()
    judge_sense = auto()
    judge_reasoning = auto()
    BestOfN_FIN = auto()
    BestOfNRW_FIN = auto()
    SOCRATES_FIN_RE_FLASH = auto()
    SOCRATES_FIN_PRE = auto()
    Intervention_CriteriaG = auto()
    SOCRATES_INF_RE_Meta_Dynamic = auto()
    THINKING_MAKER = auto()

async def async_print(*args, **kwargs):
    await asyncio.to_thread(print, *args, **kwargs)
class Duplicater:
    def __init__(self, tokenizer_path="/data/jcxy/llm_model/Qwen2-7B-Instruct-AWQ", sensitive_words='', iteration=0):
        # Load tokenizer
        self.tokenizer_path = tokenizer_path
        self.current_score = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.tokenizer_path.find("PsyLLM") != -1 or self.tokenizer_path.find("Qwen") != -1 or self.tokenizer_path.find("xiaoyun") != -1:
            self.tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
            self.tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
            self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        self.sensitive_words = sensitive_words  # Ensure it's a list of words
        if isinstance(self.sensitive_words, str):
            self.sensitive_words = self.sensitive_words.split()
        self.iteration = iteration
        self.llm = ""
        self.llm2 = ""
      
        self.llm_dict = {}
        self.threshold = 0

    def detect_duplicates(self, text):
        # Encode text using tokenizer
        encoded_text = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        unique_elements, counts = torch.unique(encoded_text, return_counts=True)
        duplicates = unique_elements[counts > 1]
        duplicate_counts = counts[counts > 1]
        return list(duplicates), list(duplicate_counts)
    
    def detect_sensitive_word(self, content):
        if any(word in content for word in self.sensitive_words):
            for word in self.sensitive_words:
                if word in content:
                    #print(f"回答内容: {content}")
                    print(f"敏感词词汇: {word}")
            return False
        return True
    
    def forward(self, content):
        if isinstance(content, list):
            concatenated_content = "".join(item["content"] for item in content)
            content = concatenated_content
        duplicates, counts = self.detect_duplicates(content)
        if counts and max(counts) > 800:
            # for word, count in zip(duplicates, counts):
            #     if count > 500:
            #         print(f"单词 '{self.tokenizer.decode([word])}' 重复了 {count} 次。")
            return False
        else:
            return True

    def get_last_key_value(self, data_dict):
        if not isinstance(data_dict, dict) or not data_dict:
            return None
        last_key = list(data_dict.keys())[-1]
        return data_dict[last_key]
    

    async def process_result(self, result, stage):
        """
        处理单个 result，根据 stage 进行不同的处理，并根据条件生成输出或记录错误。

        :param result: 要处理的结果
        :param stage: 当前处理阶段
        :return: 处理后的结果或 None
        """
        duplicate_mistake = 0
        sensitive_word_mistake = 0
        length_mistake = 0
        cut_length = 0

        if result == "skip":
            return None
        
        if isinstance(result, str):
            return None
            
        # 获取 val，根据不同的 stage 进行处理
        if any(keyword in str(stage) for keyword in ["LLM_EXPLORE", "MCTSR", "META_JUSTICE", "MorphicResonanceKernel"]):
            val = result.get("chosen")
        elif "Evol_Instruct_noA" in str(stage):
            prompts = result.get("prompt", [])
            if prompts and isinstance(prompts[-1], dict):
                val = prompts[-1].get("content", "")
            else:
                val = ""
        elif "REJECTED" in str(stage):
            val = result.get("rejected", "")
        elif "chosen" in result:
            #if result["chosen"]:
            val = result.get("chosen", "")
            #else:
                #val = result.get("prompt", "")
            

        elif "output" in result:
            if result["output"]:
                val = result.get("output", "")
            else:
                val = result.get("instruction", "")
        else:
            val = ""
        # 如果 val 是列表，拼接其中的内容
        if isinstance(val, list):
            val = "".join(item.get('content', '') for item in val if isinstance(item, dict))

        # 进行唯一性检查
        try:
            is_unique = self.forward(val)
        except Exception as e:
            print("Error during uniqueness check:", val, e)
            is_unique = False  # 如果出现异常，将其视为非唯一

        # 检测敏感词
        #if "REJECTED" not in str(stage):
        has_sensitive = self.detect_sensitive_word(val)
        

        # 检查长度
        is_length_ok = len(val) >= cut_length

        # 根据条件执行相应的操作
        if not is_unique and "QA_REJECTED" not in str(stage):
            duplicate_mistake += 1
        elif not has_sensitive:
            sensitive_word_mistake += 1
        elif not is_length_ok:
            length_mistake += 1
        elif is_unique and has_sensitive and is_length_ok:
            # 满足所有条件，返回结果
            if isinstance(result, list):
                return result[0]
            else:
                return result

        # 检查是否有需要修复的项，并输出统计信息
        if duplicate_mistake != 0 or sensitive_word_mistake != 0 or length_mistake != 0:
            print(f"\n\n\n{stage} 遇到重复输出 {duplicate_mistake} 条, 敏感词输出 {sensitive_word_mistake} 条, 长度不足输出 {length_mistake} 条 fix....\n\n")
            print(val)
            return None

    
    async def process_and_filter_async(self, processer: Process_Controller, inputs, stage, max_retries=3, cut_length=1, rollout=4):
        """
        异步处理和过滤输入数据，根据不同的处理阶段生成结果。

        Args:
            processer: 处理器实例。
            inputs (list): 输入数据列表。
            stage (ProcessStage): 当前处理阶段。
            max_retries (int): 最大重试次数。

        Yields:
            dict: 处理后的结果或重试请求。
        """
        fix_cache = copy.deepcopy(inputs)  # 确保输入数据的独立性
        process_stages = {
            ProcessStage.Scenario_to_Query: processer.process_stage_Scenario_to_Query,
            ProcessStage.ASYNC_SCENARIO_TO_REPORT_IMPEDANCE: processer.async_process_stage_reverse_Report,
            ProcessStage.LLM_EXPLORE_ASYNC: processer.async_process_llm_explore,
            ProcessStage.LLM_EXPLORE_ASYNC_GPT: processer.async_process_llm_explore,
            ProcessStage.LLM_EXPLORE_ASYNC2: processer.async_process_llm_explore,
            ProcessStage.PsyEval_IMPEDANCE: processer.process_stage_PsyEval_impedance,
            ProcessStage.REWARD_THERAPY_QUALITY: processer.process_stage_reward_therapy_quality,
            ProcessStage.CONVERTFOREVAL: processer.process_stage_convertforEval,
            ProcessStage.IDENTITY_GENERATE: processer.process_stage_identity_generate,
            ProcessStage.IDENTITY_REJECTED: processer.process_stage_identity_rejected,
            ProcessStage.QA_REJECTED: processer.process_stage_qa_rejected,
            ProcessStage.DIALOG_REJECTED: processer.process_stage_dialog_rejected,
            ProcessStage.EVOL_INSTRUCT: processer.Evol_Instruct,
            ProcessStage.EVOL_INSTRUCT_AME: processer.Evol_Instruct_AME,
            ProcessStage.ROLE_GENERATE: processer.process_stage_roleplay_generate,
            ProcessStage.GenesisRAM: processer.process_stage_genesis_ram,
            ProcessStage.Faynman_Technique: processer.Faynman_Technique,
            ProcessStage.REWRITE_INSTRUCT: processer.process_stage_rewrite_instruct,
            ProcessStage.REPORT_CLASS: processer.process_stage_report_class,
            ProcessStage.DIALOG_GENERATE: processer.process_stage_dialog_generate,
            ProcessStage.Evol_Instruct_noA: processer.Evol_Instruct_noA,
            ProcessStage.EVOL_MCTSR: processer.async_process_llm_explore_evol,
            ProcessStage.Intervent: processer.async_process_Intervent,
            ProcessStage.LLMExplorer_Zero_psy: processer.async_process_llm_explore_psy,
            ProcessStage.SOCRATES_INF: processer.async_process_llm_explore_Socrates_infinite,
            ProcessStage.SOCRATES_FIN: processer.async_process_llm_explore_Socrates_finite,
            ProcessStage.SOCRATES_FIN_PRE: processer.async_process_llm_explore_Socrates_finite,
            ProcessStage.SOCRATES_INF_RE: processer.async_process_llm_explore_Socrates_infinite,
            ProcessStage.SOCRATES_FIN_RE: processer.async_process_llm_explore_Socrates_finite,
            ProcessStage.SOCRATES_INF_RE_Meta: processer.async_process_llm_explore_Socrates_infinite,
            ProcessStage.SOCRATES_INF_RE_Meta_THINK: processer.async_process_llm_explore_Socrates_infinite,
            ProcessStage.SOCRATES_FIN_RE_Meta: processer.async_process_llm_explore_Socrates_finite,
            ProcessStage.SOCRATES_FIN_RE_Meta_THINK: processer.async_process_llm_explore_Socrates_finite,
            ProcessStage.SOCRATES_FIN_RE_FLASH: processer.async_process_llm_explore_Socrates_finite,
            ProcessStage.SOCRATES_INF_RE_Meta_Dynamic: processer.async_process_llm_explore_Socrates_dynamic_finite,
            ProcessStage.THINKING_MAKER: processer.produce_thinking,
            ProcessStage.SOCRATES_RULE_INF: processer.async_process_llm_explore_Socrates_Rule_infinite,
            ProcessStage.SOCRATES_RULE_FIN: processer.async_process_llm_explore_Socrates_Rule_finite,
            ProcessStage.META_JUSTICE: processer.async_process_llm_explore_meta_justice,
            ProcessStage.MATH_TRANSLATION: processer.expert_math_translation,
            ProcessStage.DIALOG_TO_THINKER: processer.convert_muldialog_to_thinker,
            ProcessStage.SOCRATES_INF_ROLE: processer.async_process_llm_explore_Socrates_infinite_roleplay,
            ProcessStage.SOCRATES_FIN_ROLE: processer.async_process_llm_explore_Socrates_finite_roleplay,
            ProcessStage.DISTILL_FIN_ROLE: processer.async_process_distill_finite_roleplay,
            ProcessStage.DISTILL_INF_ROLE: processer.async_process_distill_infinite_roleplay,
            ProcessStage.COUNREPORT_INF: processer.async_process_CounReport_infinite,
            ProcessStage.CharacterV1_to_V3: processer.CharacterV1_to_V3,
            ProcessStage.TextGrad_FIN: processer.async_process_TextGrad_finite,
            ProcessStage.TextGrad_INF: processer.async_process_TextGrad_infinite,
            ProcessStage.SYNTHETIC_SYSTEM: processer.async_process_synthetic_system,
            ProcessStage.PSY_REJECTED: processer.process_stage_psy_rejected,
            ProcessStage.DISTILL_INF: processer.async_process_Distill_infinite,
            ProcessStage.PsyTextGrad_FIN: processer.PsyTextGrad_finite,
            ProcessStage.PsyTextGrad_INF: processer.PsyTextGrad_infinite,
            ProcessStage.BestOfN_FIN: processer.async_process_BestOfN_finite,
            ProcessStage.BestOfNRW_FIN: processer.BestOfNRW_finite,
            ProcessStage.MorphicResonanceKernel: processer.MorphicResonanceKernel_infinite,
            ProcessStage.JUDGE_ANALOGY: processer.judge_analogy,
            ProcessStage.REWARD_SCORE: processer.Reward_Score,
            ProcessStage.PROMPTREWARD_SCORE: processer.PromptReward_Score,
            ProcessStage.TextGrad_Catalyst: processer.TextGrad_Catalyst,
            ProcessStage.Rewrite_Dialog: processer.Rewrite_Dialog,
            ProcessStage.Endless_Therapy_Dialog: processer.endless_therapy_dialog,
            ProcessStage.Endless_Therapy_Answer: processer.endless_therapy_answer,
            ProcessStage.PARALLEL_EVALUATION: processer.async_process_parallel_evalution,
            ProcessStage.JUDGE_QUANTITY: processer.process_Judge_Quantity,
            ProcessStage.judge_fabricationDetected: processer.async_process_judge_fabricationDetected,
            ProcessStage.judge_chatAbility: processer.async_process_judge_chatAbility,
            ProcessStage.judge_consultantActiveEndingDetected: processer.judge_consultantActiveEndingDetected,
            ProcessStage.Rewrite_fabricate: processer.async_process_Rewrite_fabricate,
            ProcessStage.judge_sense: processer.judge_sense,
            ProcessStage.judge_reasoning: processer.async_process_judge_YNReasoning,
            ProcessStage.Intervention_CriteriaG: processer.Intervention_CriteriaG_finite

            # 添加其他阶段的处理函数
        }
        process_stage_func = process_stages.get(stage)
        if stage == ProcessStage.PARALLEL_EVALUATION:
            explorers = [
                LLMExplorer_Socrates(
                    llm=value,
                    api_llm=value,
                    initial_threshold=self.threshold,
                    current_score = self.current_score,
                    max_iter=4
                )
                for _,value in self.llm_dict.items()
            ]
            tasks = [process_stage_func(explorers, fix_cache[0])]
        elif stage == ProcessStage.LLM_EXPLORE_ASYNC:
            explorers = [
                LLMExplorer(
                    llm=self.llm,
                    api_llm=self.llm2,
                    initial_threshold=self.threshold,
                    current_score = self.current_score,
                    max_iter=100000000
                )
                for _ in fix_cache
            ]
            tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
        elif stage == ProcessStage.LLM_EXPLORE_ASYNC2:
            explorers = [
                LLMExplorer2(
                    llm=self.llm,
                    api_llm=self.llm2,
                    initial_threshold=self.threshold,
                    current_score = self.current_score,
                    max_iter=100000000
                )
                for _ in fix_cache
            ]
            tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
        elif stage == ProcessStage.LLM_EXPLORE_ASYNC_GPT:
            explorers = [
                LLMExplorer_GPT(
                    llm=self.llm,
                    api_llm=self.llm2,
                    initial_threshold=self.threshold,
                    current_score = self.current_score,
                    max_iter=100000000
                )
                for _ in fix_cache
            ]
            tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
        elif stage == ProcessStage.LLMExplorer_Zero or stage == ProcessStage.EVOL_MCTSR:
            explorers = [
                LLMExplorer_Zero(
                    llm=self.llm,
                    initial_threshold=self.threshold,
                    max_iter=100000000
                )
                for _ in fix_cache
            ]
            tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
        elif stage == ProcessStage.LLMExplorer_Zero_psy:
            explorers = [
                LLMExplorer_Zero_psy(
                    llm=self.llm,
                    initial_threshold=self.threshold,
                    max_iter=100000000
                )
                for _ in fix_cache
            ]
            tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
        #elif stage == ProcessStage.SOCRATES_INF or stage == ProcessStage.SOCRATES_INF_RE or stage == ProcessStage.SOCRATES_FIN_RE or stage == ProcessStage.SOCRATES_FIN or stage == ProcessStage.SOCRATES_INF_ROLE:
        elif "SOCRATES" in str(stage):
            if stage == ProcessStage.SOCRATES_FIN_PRE:
                print("MCTSV2")
                from Swimming_Pool_Async.LLMExplorer_Socrates_Pre import LLMExplorer_Socrates
                explorers = [
                    LLMExplorer_Socrates(
                        llm=self.llm,
                        api_llm=self.llm2,
                        initial_threshold=self.threshold,
                        max_iter=rollout
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.SOCRATES_INF_RE or stage == ProcessStage.SOCRATES_FIN_RE:
                from Swimming_Pool_Async.LLMExplorer_Socrates_re import LLMExplorer_Socrates
                explorers = [
                    LLMExplorer_Socrates(
                        llm=self.llm,
                        api_llm=self.llm2,
                        initial_threshold=self.threshold,
                        max_iter=rollout
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.SOCRATES_INF_RE_Meta or stage == ProcessStage.SOCRATES_FIN_RE_Meta:
                print("MCTSV3")
                from Swimming_Pool_Async.LLMExplorer_Socrates_re_Meta import LLMExplorer_Socrates
                explorers = [
                    LLMExplorer_Socrates(
                        llm=self.llm,
                        api_llm=self.llm2,
                        initial_threshold=self.threshold,
                        max_iter=rollout
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.SOCRATES_INF_RE_Meta_THINK or stage == ProcessStage.SOCRATES_FIN_RE_Meta_THINK:
                print("MCTSV3")
                from Swimming_Pool_Async.LLMExplorer_Socrates_re_Meta_think import LLMExplorer_Socrates
                explorers = [
                    LLMExplorer_Socrates(
                        llm=self.llm,
                        api_llm=self.llm2,
                        initial_threshold=self.threshold,
                        max_iter=rollout
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.SOCRATES_FIN_RE_FLASH:
                print("MCTSV3")
                from Swimming_Pool_Async.LLMExplorer_Socrates_re_Meta_flash import LLMExplorer_Socrates
                explorers = [
                    LLMExplorer_Socrates(
                        llm=self.llm,
                        api_llm=self.llm2,
                        initial_threshold=self.threshold,
                        max_iter=rollout
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.SOCRATES_INF_RE_Meta_Dynamic:
                print("MCTSV3_Dynamic")
                from Swimming_Pool_Async.LLMExplorer_Socrates_re_Meta_dynamic import LLMExplorer_Socrates
                explorers = [
                    LLMExplorer_Socrates(
                        llm=self.llm,
                        api_llm=self.llm2,
                        initial_threshold=self.threshold,
                        max_iter=rollout
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.SOCRATES_RULE_INF or stage == ProcessStage.SOCRATES_RULE_FIN:
                explorers = [
                    LLMExplorer_Socrates_Rule(
                        llm=self.llm,
                        api_llm=self.llm,
                        initial_threshold=self.threshold,
                        max_iter=4
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.MorphicResonanceKernel:
                explorers = [
                    MorphicResonanceKernel(
                        llm=self.llm,
                        api_llm=self.llm,
                        initial_threshold=self.threshold,
                        max_iter=8
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.SOCRATES_INF_ROLE:
                explorers = [
                    LLMExplorer_Socrates_roleplay(
                        llm=self.llm,
                        api_llm=self.llm,
                        initial_threshold=self.threshold,
                        max_iter=3
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]
            elif stage == ProcessStage.META_JUSTICE:
                explorers2 = [
                    LLMExplorer_Socrates_Judge(
                        llm=self.llm,
                        api_llm=self.llm,
                        initial_threshold=self.threshold,
                        max_iter=4
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer2, item) for explorer2, item in zip(explorers2, fix_cache)]
            else:
                print("use_MCTSV1")
                from Swimming_Pool_Async.LLMExplorer_Socrates import LLMExplorer_Socrates
                explorers = [
                    LLMExplorer_Socrates(
                        llm=self.llm,
                        api_llm=self.llm2,
                        initial_threshold=self.threshold,
                        max_iter=rollout
                    )
                    for _ in fix_cache
                ]
                tasks = [process_stage_func(explorer, item) for explorer, item in zip(explorers, fix_cache)]

        elif stage == ProcessStage.REWARD_THERAPY_QUALITY:
            tasks = [process_stage_func(item, update_prompt, use_intervent) for (item, update_prompt, use_intervent) in fix_cache]
        elif "TextGrad" in str(stage):
            tasks = [process_stage_func(item, n=rollout) for item in fix_cache]
        else:
            tasks = [process_stage_func(item) for item in fix_cache]

        # 使用 asyncio.as_completed 来并发处理任务
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            if isinstance(result, bool):
                if result == True:
                    yield True
                elif result == False:
                    yield False
            if isinstance(result, str):
                if result == "<|_error_|>":
                    yield None
            else:
                if "reward_records" in result:
                    results = [result["gene_records"],result["reward_records"]]
                
                if isinstance(result, list):
                    results = result
                else:
                    results = [result]
                if "SOCRATES" in str(stage) or "MorphicResonanceKernel" in str(stage):
                    results1, results2 = results[0], results[1]
                    if results1:
                        for result in results1:
                            result = await self.process_result(result, stage)
                            yield [[result], 1]
                    else:
                        yield [None, 1]
                    if results2:
                        for result in results2:
                            result = await self.process_result(result, stage)
                            yield [[result], 2]
                    else:
                        yield [None, 2]
                else:
                    result = await self.process_result(result, stage)
                    yield result
                
