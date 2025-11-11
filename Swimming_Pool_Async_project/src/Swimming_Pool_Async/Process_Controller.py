import asyncio
import copy
import json
import logging
import re
import traceback
from typing import Optional

from .LLM_Core import LLM_Core
from .Prompter import Prompter
from .Tools import Tools

class Process_Controller:
    def __init__(self, llm: LLM_Core, tools: Tools):
        """
        Inicializa el controlador de procesos con los componentes necesarios.
        Se han eliminado todos los atributos y componentes no utilizados por LLMExplorer_Socrates.
        """
        self.llm = llm
        self.tools = tools
        self.prompter = Prompter()
        self.logger = logging.getLogger(__name__)
        
        # Plantilla de datos base para las llamadas a la API del LLM.
        self.data_template = {
            "model": self.llm.api_model,
            "messages": [],
            "temperature": 0.95,
            "top_p": 0.9,
            "extra_body": {},
            "stream": False,
        }

    async def receive_data(self, llm: LLM_Core, data, max_retries=1, initial_delay=1):
        """
        Recibe datos de forma asíncrona del modelo LLM con reintentos en caso de error.
        """
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay
        
        while attempt < max_retries:
            try:
                async for chunk in llm.async_model(data=data_copy):
                    json_string = chunk
                    chunk_data = json.loads(json_string)
                    if data_copy.get('stream', False):
                        output += chunk_data['choices'][0]['delta']['content']
                    else:
                        output += chunk_data['choices'][0]['message']['content']
                return output
            except Exception as e:
                error_message = traceback.format_exc()
                self.logger.error(f"Error al recibir datos: {error_message}")
                attempt += 1
                if attempt < max_retries:
                    self.logger.info(f"Reintentando la solicitud en {delay}s... ({attempt}/{max_retries})")
                    await asyncio.sleep(delay)
                    delay *= 1.1
                else:
                    self.logger.error("Se ha superado el número máximo de reintentos. La solicitud ha fallado.")
                    break
        return output if output else "<|_error_|>"

    def check_contains_sensitive(self, string_a: str) -> bool:
        """
        Comprueba si una cadena de texto contiene alguna de las palabras sensibles definidas en las herramientas.
        """
        string_list = self.tools.sensitive_words
        for item in string_list:
            if item in string_a:
                return True
        return False

    def _extract_content(self, text: str, pattern: str) -> Optional[str]:
        """
        Extrae contenido de un texto utilizando un patrón de regex.
        """
        if not text:
            return None
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_dialog(self, text: str, pattern: str) -> Optional[str]:
        """
        Extrae y formatea un diálogo de múltiples turnos a partir de un texto.
        """
        dialog_text = self._extract_content(text, pattern)
        if not dialog_text:
            return None
            
        dialog_list = self.tools.parse_fields(dialog_text)
        
        first_user_index = next((i for i, item in enumerate(dialog_list) if item["role"] == "user"), None)
        if first_user_index is not None:
            dialog_list = dialog_list[first_user_index:]
        
        formatted_dialog = ""
        for item in dialog_list:
            role_name = "来访者" if item["role"] == "user" else "咨询师"
            formatted_dialog += f"{role_name}：" + item["content"].strip() + "\n"
                
        return formatted_dialog

    def extract_total_score(self, assessment_text: str) -> Optional[float]:
        """
        Extrae la puntuación total del texto de evaluación de forma robusta.
        """
        if not isinstance(assessment_text, str) or not assessment_text:
            return None

        keywords = ['Total Score', 'Final Score', 'Overall Score', '总评分', '综合得分', '最终得分', 'Score', '得分']
        keyword_pattern = '|'.join(keywords)
        number_pattern = r'(-?\d+(?:\.\d+)?)'
        
        flexible_pattern = re.compile(
            f'({keyword_pattern})'
            r'[\s\S]{0,50}?'
            f'({number_pattern})',
            flags=re.IGNORECASE
        )
        
        matches = flexible_pattern.findall(assessment_text)
        
        if matches:
            last_match = matches[-1]
            score_str = last_match[-1]
            try:
                return float(score_str)
            except (ValueError, TypeError):
                pass
        
        return None

    async def Generate_Response(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=None):
        """
        Genera una respuesta estándar del LLM.
        """
        for attempt in range(int(max_retries)):
            data_template2 = copy.deepcopy(data_template)
            Example_Response = await self.receive_data(choose_llm, data_template2)
            
            if self.check_contains_sensitive(Example_Response):
                self.logger.warning(f"Respuesta contiene palabras sensibles. Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue

            if pattern:
                Example_Response = self._extract_content(Example_Response, pattern)
                if Example_Response is None:
                    self.logger.warning(f"No se pudo encontrar el patrón en la respuesta. Reintentando ({attempt + 1}/{max_retries}).")
                    continue
            
            return Example_Response, "" # Devuelve tupla para consistencia
        return "<|_error_|>", ""

    async def Generate_EnhanceResponse(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=None):
        """
        Genera una respuesta mejorada del LLM.
        """
        for attempt in range(max_retries):
            template_copy = copy.deepcopy(data_template)
            template_copy["model"] = choose_llm.api_model
            response = await self.receive_data(choose_llm, template_copy)
            
            if pattern:
                Example_Response = self._extract_content(response, pattern)
                if Example_Response is None:
                    self.logger.warning(f"No se pudo encontrar el patrón en la respuesta mejorada. Reintentando ({attempt + 1}/{max_retries}).")
                    continue
            else:
                Example_Response = response
            
            if self.check_contains_sensitive(Example_Response):
                self.logger.warning(f"Respuesta mejorada contiene palabras sensibles. Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue

            return Example_Response
        return "<|_error_|>"

    async def Generate_PsyResponse(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=r'#+\s*多轮对话\s*#+(.*)'):
        """
        Genera una respuesta de diálogo psicológico.
        """
        for attempt in range(int(max_retries)):
            template_copy = copy.deepcopy(data_template)
            template_copy["model"] = choose_llm.api_model
            response = await self.receive_data(choose_llm, template_copy)
            response = response.replace("记住，", "").replace("听起来", "").replace("记得，", "")
            
            formatted_dialogue = self._extract_dialog(response, pattern)
            if not formatted_dialogue:
                self.logger.warning(f"No se pudo extraer el diálogo psicológico. Reintentando ({attempt + 1}/{max_retries}).")
                continue
            
            token_count = len(self.llm.tokenizer.encode(formatted_dialogue))
            if token_count < 800 or self.check_contains_sensitive(formatted_dialogue):
                self.logger.warning(f"Respuesta Psy no válida (longitud: {token_count} o sensible). Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue
            
            return formatted_dialogue
        return "<|_error_|>"

    async def Generate_EnhancePsyResponse(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=r'#+\s*强对话\s*#+(.*)'):
        """
        Genera una respuesta de diálogo psicológico mejorada.
        """
        for attempt in range(int(max_retries)):
            template_copy = copy.deepcopy(data_template)
            template_copy["model"] = choose_llm.api_model
            response = await self.receive_data(choose_llm, template_copy)
            response = response.replace("记住，", "").replace("听起来", "").replace("记得，", "")

            formatted_dialogue = self._extract_dialog(response, pattern)
            if not formatted_dialogue:
                self.logger.warning(f"No se pudo extraer el diálogo Psy mejorado. Reintentando ({attempt + 1}/{max_retries}).")
                continue

            token_count = len(self.llm.tokenizer.encode(formatted_dialogue))
            if token_count < 800 or self.check_contains_sensitive(formatted_dialogue):
                self.logger.warning(f"Respuesta Psy mejorada no válida (longitud: {token_count} o sensible). Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue
            
            return formatted_dialogue
        return "<|_error_|>"
        
    async def Judge_Quantity(self, choose_llm: LLM_Core, data_template, max_retry=3, refine_pattern=r'#+\s*改进建议\s*#*(.*)'):
        """
        Evalúa una respuesta, extrayendo puntuación, justificación y sugerencias de mejora.
        """
        data_template2 = copy.deepcopy(data_template)
        data_template2["model"] = choose_llm.api_model
        
        for attempt in range(max_retry):
            judge_text = await self.receive_data(choose_llm, data_template2)
            
            score = self.extract_total_score(judge_text)
            refine = self._extract_content(judge_text, refine_pattern) if refine_pattern else "wu"
            
            if score is None or refine is None or not (0 <= score <= 10):
                self.logger.warning(f"Fallo en la evaluación (score: {score}). Reintentando ({attempt + 1}/{max_retry}).\nRespuesta: {judge_text}")
                continue
                
            return score, judge_text
            
        return 3.0, ""