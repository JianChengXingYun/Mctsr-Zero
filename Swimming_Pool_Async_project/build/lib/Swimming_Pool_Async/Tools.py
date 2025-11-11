from collections import defaultdict
import json
import os
import random
import re
import uuid
class Tools:
    def __init__(self, filename, tokenizer):
        self.filename = filename
        self.tokenizer = tokenizer
        self.filename2 = ""
        self.filename3 = ""
        self.input_data = []
        self.sensitive_words = []
        
    def deduplicate_by_context(self, data):
        unique_contexts = {}
        deduplicated_data = []
        for item in data:
            line = item
            if not line:  # 跳过空行
                continue
            try:
                if not isinstance(item, dict):
                    input_data = json.loads(line)
                else:
                    input_data = item
            except json.JSONDecodeError as e:
                print(f"JSON解析失败，跳过该行：{line}\n错误：{e}")
                continue
            if not input_data["chosen"]:
                continue
            # 遍历可能的字段
            for field in ['context', 'instruction', 'prompt','chosen', 'output']:
                if field in input_data and input_data[field] and input_data[field]!=[]:
                    break
            else:
                # 如果没有找到任何指定的字段，则跳过该条数据
                continue

            # 如果字段是列表，则取列表中最后一个元素的 "content"
            if isinstance(input_data[field], list):
                # 确保列表不为空且每个元素是字典并含有 "content"
                if input_data[field] and isinstance(input_data[field][-1], dict) and "content" in input_data[field][-1]:
                    context_str = input_data[field][-1]["content"]
                else:
                    continue
            else:
                context_str = input_data[field]
            # 添加去重判断
            if context_str and context_str not in unique_contexts:
                unique_contexts[context_str] = True
                deduplicated_data.append(input_data)

        return deduplicated_data

    
    def process_context(self, data):
        out_list = []
        for item in data:
            input_data = json.loads(item.strip())
            processed_data = {}

            context_set = False
            for field in ['label','context', 'prompt', 'instruction', "Counseling_Report"]:
                if field in input_data and input_data[field]:
                    if isinstance(input_data[field], list):
                        input_data[field] = input_data[field][-1]["content"]
                    processed_data['context'] = input_data[field]
                    context_set = True
                    break

            if not context_set:
                try:
                    for key, value in input_data.items():
                        if isinstance(value, list) and value:
                            processed_data['context'] = self.taken_label(value)
                            context_set = True
                            break
                        elif isinstance(value, str) and value:
                            processed_data['context'] = value
                            context_set = True
                            break
                except:
                    print(input_data)
                    

            # 将剩余的字段添加到 processed_data 中
            for key, value in input_data.items():
                if key not in processed_data:
                    processed_data[key] = value

            out_list.append(processed_data)

        return out_list
    

    def Read_Document_PsyEval(self):
        self.input_data = []
        existing_contexts = set()
        skipped_count = 0
        processed_count = 0
        primary_types = defaultdict(list)
        # Check if file exists and read the data. If not, start from beginning.
        if os.path.exists(self.filename2):
            with open(self.filename2, 'r', encoding='utf-8') as file:
                existing_data = file.readlines()
                existing_data = self.process_context(existing_data)
                existing_data = self.deduplicate_by_context(existing_data)
                #print()
                processed_count = len(existing_data)
                for item in existing_data:
                    if 'context' in item:
                        context = item['context']
                    else:
                        context = self.taken_label(item.get("chosen", []))
                    existing_contexts.add(context)

        with open(self.filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
            lines = self.process_context(lines)
            lines = self.deduplicate_by_context(lines)
            total_lines = len(lines)

            for item in lines:
                if item['context'] in existing_contexts:
                    skipped_count += 1
                    continue

                existing_contexts.add(item['context'])
                
                # 提取 Primary Type
                counseling_report = item.get('Counseling_Report', '')
                primary_type = self.extract_primary_type(counseling_report)
                
                if primary_type:
                    primary_types[primary_type].append(item)

            # 选择16种不同的 Primary Type，每种最多10个
        selected_data = self.select_primary_types(primary_types)

        print(f"Total lines count: {total_lines}")
        print(f"Skipped count: {skipped_count}")
        print(f"Selected Primary Types: {len(selected_data)}")
        #print(f"Total selected items: {len(selected_data)")

        # # 添加更详细的输出
        # for type, items in selected_data.items():
        #     print(f"Primary Type: {type}, Selected items: {len(items)}")

        return selected_data, len(selected_data), processed_count
    def Read_Document_PsyEval(self,num=32):
        self.input_data = []
        existing_contexts = set()
        skipped_count = 0
        processed_count = 0
        primary_types = defaultdict(list)
        # Check if file exists and read the data. If not, start from beginning.
        if os.path.exists(self.filename2):
            with open(self.filename2, 'r', encoding='utf-8') as file:
                existing_data = file.readlines()
                #existing_data = self.process_context(existing_data)
                #existing_data = self.deduplicate_by_context(existing_data)
                processed_count = len(existing_data)
                for item in existing_data:
                    if isinstance(item, str):
                        item = json.loads(item)
       
                    existing_contexts.add(item["prompt"][-1]["content"])

        with open(self.filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
            content_list = []
            #lines = self.process_context(lines)
            #lines = self.deduplicate_by_context(lines)
            type_use = False
            total_lines = len(lines)
            print(f"Total lines count: {len(lines)}")
            for item in lines:
                if isinstance(item, str):
                    item = json.loads(item)
                
                if item["prompt"][-1]["content"] in existing_contexts:
                    skipped_count += 1
                    continue
                content_list.append(item)
                existing_contexts.add(item["prompt"][-1]["content"])
                # 提取 Primary Type
                if "class_tag" in item:
                    type_use = True
                    primary_type = item["class_tag"]
                    primary_types[primary_type].append(item)
            # 选择16种不同的 Primary Type，每种最多10个
        if type_use:
            selected_data = self.select_primary_types(primary_types,num=num)

            print(f"Total lines count: {total_lines}")
            print(f"Skipped count: {skipped_count}")
            print(f"Selected Primary Types: {len(selected_data)}")
            #print(f"Total selected items: {len(selected_data)")
            # # 添加更详细的输出
            # for type, items in selected_data.items():
            #     print(f"Primary Type: {type}, Selected items: {len(items)}")
            return selected_data, len(selected_data), processed_count
        else:
            return content_list[:num*16], len(content_list[:num*16]), processed_count
    def extract_primary_type(self, counseling_report):
        # 这里需要根据实际的 Counseling_Report 格式来提取 Primary Type
        # 这只是一个示例，你可能需要根据实际情况调整
        start = counseling_report.find("Primary Type:")
        if start != -1:
            end = counseling_report.find("\n", start)
            if end != -1:
                return counseling_report[start+13:end].strip()
        return None

    def select_primary_types(self, primary_types,num=32):
        selected_items = []
        
        # 过滤出至少有10条数据的类型
        filtered_types = [typ for typ in primary_types if len(primary_types[typ]) >= 10]
        
        if not filtered_types:
            print("No types with at least 10 items available.")
            return []
        
        # 按类型数量降序排序
        sorted_types = sorted(filtered_types, key=lambda x: -len(primary_types[x]))
        
        # 选择数量最多的前16类
        selected_types = sorted_types[:16]
        print(f"Selected top {len(selected_types)} primary types (each has ≥10 items).")
        
        # 从每类中抽取10条数据
        for typ in selected_types:
            items = primary_types[typ]
            selected = random.sample(items, num)  # 直接抽10条（已确保≥10条）
            selected_items.extend(selected)
        
        # 统计每类数量
        type_counts = {}
        for item in selected_items:
            typ = item['class_tag']
            type_counts[typ] = type_counts.get(typ, 0) + 1
        
        # 打印结果
        for typ, count in type_counts.items():
            print(f"Type: {typ}, Selected: {count}")
        
        print(f"Total selected: {len(selected_items)} (Target: 160)")
        return selected_items

    def Read_Document_01(self):
        self.input_data = []
        existing_contexts = []
        skipped_count = 0
        processed_count = 0
        # Check if file exists and read the data. If not, start from beginning.
        if os.path.exists(self.filename2):
            print("读文件：",self.filename2)
            with open(self.filename2, 'r', encoding='utf-8') as file:
                existing_data = file.readlines()
                processed_count = len(existing_data)
                for item in existing_data:
                    if not isinstance(item, dict):
                        input_data = json.loads(item.strip())
                    else:
                        input_data = item
                    if "uid" in input_data:
                        existing_contexts.append(input_data["uid"])
                    elif len(input_data["prompt"]) >= 2:
                        existing_contexts.append(input_data["prompt"][-1]["content"])
                    elif "chosen" in input_data:
                        concatenated_content = "\n".join([item["content"] for item in input_data["chosen"]])
                        existing_contexts.append(concatenated_content)
                    elif "instruction" in input_data:
                        existing_contexts.append(input_data["instruction"][-1]["content"])

        with open(self.filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
            random.shuffle(lines)
            total_lines = len(lines)
            for item in lines:
                if not isinstance(item, dict):
                    input_data = json.loads(item.strip())
                else:
                    input_data = item
                if "uid" not in input_data:
                    uid1 = uuid.uuid4()
                    uid_str = str(uid1)
                    input_data["uid"] = uid_str
                elif len(input_data["prompt"]) >= 2:
                    if input_data["prompt"][-1]["content"] in existing_contexts:
                        skipped_count += 1
                        continue
                elif "chosen" in input_data:
                    concatenated_content = "\n".join([item["content"] for item in input_data["chosen"]])
                    if concatenated_content in existing_contexts:
                        skipped_count += 1
                        continue
                elif "instruction" in input_data:
                    if input_data["instruction"][-1]["content"] in existing_contexts:
                        skipped_count += 1
                        continue
                self.input_data.append(input_data)
        processed_count = len(existing_contexts)
        print(f"total_lines count: {total_lines}")
        print(f"Skipped count: {skipped_count}")
        print(f"input_data length: {len(self.input_data)}")
        return self.input_data, total_lines, processed_count
    
    
    def parse_fields(self,data_string):
        data_string = re.sub(r'[\(（].*?[\)）]', '', data_string)
        data_string = re.sub(r'[\((].*?[\))]', '', data_string)
        data_string =(
        str(data_string)
        .replace("：",":")
        .replace("*","")
        .replace("-","")
        .replace(":\n",":")
        .replace("】","")
        .replace("【","")
        .replace("用户:","来访者:")
        .replace("Assistant:","assistant:")
        .replace("User:","user:")
        .replace("assistant\n:","assistant:")
        .replace("user\n:","user:")
        .replace("南希:，","南希:")
        .replace("咨询师:，","咨询师:")
        .replace("。，","。")
        .replace(".,",".")
        .replace("“","")
        .replace("”","")
        )
        data_string = re.sub(r"(来访者:|咨询师:)\s*\n", r"\1", data_string)
        data_string = re.sub(r"(来访者:|南希:)\s*\n", r"\1", data_string)
        data_string = re.sub(r"(user:|assistant:)\s*\n", r"\1", data_string)
        # 去除注释和空行
        lines = [line for line in data_string.split('\n') if line]
        # 解析字段
        sum_list = []
        for line in lines:
            line = line.strip()
            if len(line)<3 or line.find(":")==-1:
                continue
            key, value = line.split(':', 1)
            key = key.strip() 
            value = value.strip()
            # 去掉value开头的逗号
            if value.startswith("，") or value.startswith(","):
                value = value[1:].strip()
                
            if key.find("用户")!=-1 or key.find("来访者")!=-1 or key.find("user")!=-1:
                cache = { "content": value.strip(), "role": "user" }
            elif key.find("分析")!=-1:
                cache = { "content": value.strip(), "role": "analysis" }
            elif key.find("咨询师")!=-1 or key.find("assistant")!=-1 or key.find("psychologist")!=-1 or key.find("南希")!=-1:#南希:
                cache = { "content": value.strip(), "role": "assistant" }
            else:
                continue
            value=value.replace(" ","").replace(":","")
            sum_list.append(cache)

        return sum_list



