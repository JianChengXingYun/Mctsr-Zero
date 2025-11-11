from transformers import AutoTokenizer

class Prompter:#，回答案例字数控制在500字以内
    def __init__(self,tokenizer_path="/data/jcxy/llm_model/Qwen2.5-7B-Instruct-AWQ"):
        # self.tokenizer_path = tokenizer_path
        # self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # if self.tokenizer_path.find("PsyLLM") != -1 or self.tokenizer_path.find("Qwen") != -1 or self.tokenizer_path.find("xiaoyun") != -1:
        #     self.tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
        #     self.tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
        #     self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
            
        # elif self.tokenizer_path.find("Llama-3") != -1:
        #     self.tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
        #     self.tokenizer.add_special_tokens({"pad_token": "<|eot_id|>"})
        # elif self.tokenizer_path.find("gemma-2") != -1 or self.tokenizer_path.find("xingyun") != -1:
        #     self.tokenizer.add_special_tokens({"bos_token": "<start_of_turn>"})
        #     self.tokenizer.add_special_tokens({"eos_token": "<end_of_turn>"})
        # bos = self.tokenizer.bos_token
        # eos = self.tokenizer.eos_token
        # #spc = 
        # if tokenizer_path.find("Qwen")!=-1 or tokenizer_path.find("Llama")!=-1:
        #     spc = "assistant"
        # elif tokenizer_path.find("gemma")!=-1:
        #     spc = "model"
        # else:
        #     spc = "assistant"

        self.emotion_recognition_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**1. Emotion Recognition Ability**
- **Accurately and sensitively identify and understand the user's emotions, including complex, mixed, or subtle emotional states**
  - **0 points:** Unable to recognize or understand the user's emotions.
  - **1 point:** Can partially recognize and understand the user's emotions, but not with enough accuracy or detail, especially in the case of complex, mixed, or subtle emotional states.
  - **2 points:** Accurately and thoroughly recognizes and understands the user's emotions, including complex, mixed, or subtle emotional states.
  
- **Express complex emotions clearly and directly in language**
  - **0 points:** Unable to express emotions clearly and directly, with language being vague or incoherent.
  - **1 point:** Can express emotions, but lacks clarity or directness, with difficulty in articulating complex emotional states.
  - **2 points:** Accurately and clearly expresses complex emotions in direct and understandable language.
  
- **Highly sensitive to subtle emotional shifts, tone, and word choices, and can quickly identify the underlying emotional core in the user's language**
  - **0 points:** Unaware of emotional shifts, tone, or word choices, unable to identify the hidden emotional core.
  - **1 point:** Shows some sensitivity to emotional shifts, tone, and word choices, occasionally identifying the underlying emotional core.
  - **2 points:** Highly sensitive to subtle emotional shifts, tone, and word choices, quickly identifying the hidden emotional core in the user's language.
  
- **Able to recognize both overt emotions and the subtle emotions that lie beneath the surface**
  - **0 points:** Only able to identify overt, basic emotions, unable to recognize subtle or hidden emotions.
  - **1 point:** Can identify overt emotions, but has limited ability to discern the subtle emotions hidden beneath the surface.
  - **2 points:** Able to recognize both overt emotions and the subtle emotions that lie beneath the surface (e.g., guilt beneath frustration or fear beneath anxiety).

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.analysis_ability_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**2. Analysis Ability**
- **Ability to Understand the Impact of Unconscious Conflicts, Emotional Drives, and Early Experiences on Current Behavior**
  - **0 points:** Unable to understand the impact of unconscious conflicts, emotional drives, and early experiences on current behavior.
  - **1 point:** Has a basic understanding of unconscious conflicts, emotional drives, and early experiences, but lacks depth in analysis.
  - **2 points:** Demonstrates a deep understanding of how unconscious conflicts, emotional drives, and early experiences influence current behavior.
  
- **Sensitivity to Hidden Information in the Client's Speech, Dreams, and Defense Mechanisms, and Ability to Interpret Using Psychodynamic Theory**
  - **0 points:** Unable to detect hidden information in the client’s speech, dreams, and defense mechanisms, and does not apply psychodynamic theory for interpretation.
  - **1 point:** Able to detect some hidden information, but limited in applying psychodynamic theory for interpretation.
  - **2 points:** Highly sensitive to hidden information in the client’s speech, dreams, and defense mechanisms, and skillfully applies psychodynamic theory for interpretation.
  
- **Ability to Quickly Identify and Accurately Analyze the Client’s Underlying Emotions and Relationship Patterns, Connecting Them to Early Experiences and Internal Conflicts**
  - **0 points:** Unable to identify the client’s underlying emotions or relationship patterns, and cannot connect them to early experiences or internal conflicts.
  - **1 point:** Able to identify some underlying emotions and relationship patterns, but analysis is imprecise and connections to early experiences and internal conflicts are not fully established.
  - **2 points:** Quickly identifies and accurately analyzes the client’s underlying emotions and relationship patterns, and effectively connects them to early experiences and internal conflicts.
  
- **Ability to Clearly Guide the Client in Recognizing the Specific Context of Their Emotions or Behaviors to Facilitate Better Reflection and Response**
  - **0 points:** Unable to guide the client in recognizing the specific context of their emotions or behaviors, and cannot help the client reflect or respond effectively.
  - **1 point:** Able to partially guide the client in recognizing the context of their emotions or behaviors, but the guidance lacks clarity and the impact is limited.
  - **2 points:** Clearly guides the client in recognizing the specific context of their emotions or behaviors, helping them reflect and respond more effectively.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.semantic_understanding_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**3. Semantic Understanding Ability**
- **Accurately Interpreting the Deep Meaning in the Client’s Words**
  - **0 points:** Fails to understand the deep meaning of the client’s words, provides surface-level responses or misinterprets the client’s intentions.
  - **1 point:** Partially understands the deep meaning of the client’s words, able to identify some emotions or intentions, but may miss key details or misinterpret some aspects.
  - **2 points:** Accurately identifies and understands the deep meaning in the client’s words, capturing emotions, underlying intentions, and subtle nuances.
  
- **Understanding Subtle Emotional Expressions**
  - **0 points:** Fails to notice the client’s emotional shifts or misinterprets the emotional meaning, responses may feel rigid or irrelevant.
  - **1 point:** Able to recognize basic emotional changes in the client, but may struggle to detect more subtle emotional shifts, leading to some misalignment in responses.
  - **2 points:** Accurately understands and responds to the client’s subtle emotional expressions, perceiving the complexity and fluctuations of emotions, with responses that are well attuned.
  
- **Accurately Paraphrasing the Client’s Expression**
  - **0 points:** Paraphrasing is inaccurate, omitting important information or misunderstanding the client’s core point.
  - **1 point:** Paraphrasing is mostly accurate, capturing the main points, but may miss some details or lack precision in wording.
  - **2 points:** Paraphrases clearly and accurately, perfectly restating the client’s points and emotions, conveying the underlying meaning of the original expression.
  
- **Guiding the Client to Reflect Further on Their Feelings and Thoughts**
  - **0 points:** Fails to guide the client in reflecting on their feelings or thoughts, providing superficial or vague feedback.
  - **1 point:** Provides some guidance for the client to reflect on their feelings or thoughts, but does not fully stimulate deeper reflection or convey complete understanding.
  - **2 points:** Effectively guides the client in deep self-reflection, helping them further explore and understand their emotions and thoughts, with responses demonstrating profound empathy and understanding.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.logical_ability_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**4. Logical Ability**
- **Clear Logic and Ability to Focus on Core Issues in Discussion**
  - **0 points:** Lacks clear logical structure, the discussion lacks a clear direction, and tends to deviate from or fail to focus on the core issue.
  - **1 point:** Logic is fairly clear, and the discussion stays focused on the core issue to some extent, though it may occasionally deviate or lose coherence.
  - **2 points:** Clear and structured logic, consistently focusing on the core issue throughout the session, with no significant deviations.
  
- **Positive Directional Guidance, Avoiding Topic Deviation**
  - **0 points:** Fails to effectively guide the client, resulting in off-topic discussions or unproductive conversations.
  - **1 point:** Provides some positive guidance, but may occasionally fail to fully prevent topic deviation or may not adjust direction in a timely manner.
  - **2 points:** Consistently maintains positive guidance, ensuring the discussion stays focused on the core issue, with no deviation from the main topic.
  
- **Ability to Grasp Key Points and Flexibly Adjust the Dialogue to Achieve Goals**
  - **0 points:** Fails to identify key points in the discussion, with awkward or irrelevant topic shifts, failing to effectively drive toward the counseling goals.
  - **1 point:** Able to identify key points, but may not be flexible enough when adjusting the conversation or may experience delays, resulting in occasional inefficiency in reaching counseling goals.
  - **2 points:** Highly skilled in identifying key points and flexibly adjusting the pace and focus of the discussion to ensure the counseling goals are successfully met.
  
- **Coherent Thought Process with Clear Structure**
  - **0 points:** Disjointed or jumping thoughts, lacking a structured approach, making it difficult for the client to understand the content of the session.
  - **1 point:** The overall flow of thought is relatively coherent, but may lack clear structure in some parts, making the information somewhat unclear or hard to follow.
  - **2 points:** The thought process is very coherent, with a clear structure, guiding the client logically through the session and helping them understand various layers of the issue.
  
- **Ensuring the Conversation is Not Disjointed or Interrupted, Allowing for a Progressive Exploration of the Issue**
  - **0 points:** The conversation lacks coherence, with frequent interruptions or disjointed shifts that prevent the client from developing a deep understanding of the issue.
  - **1 point:** The conversation is mostly coherent, but may occasionally shift or break, and some issues are not fully explored or discussed in sufficient depth.
  - **2 points:** The conversation is always coherent, with no interruptions or abrupt shifts, allowing for a progressive exploration of the issue, helping the client develop a deeper understanding.
  
- **Using Logical Reasoning to Help the Client Understand the Causal Relationships Between Emotions, Behaviors, and Events**
  - **0 points:** Fails to help the client understand the causal relationships between emotions, behaviors, and events, with insufficient logical reasoning.
  - **1 point:** Provides partial help in understanding the causal relationships, but the reasoning lacks clarity or depth.
  - **2 points:** Uses clear and structured logical reasoning to help the client deeply understand the causal relationships between emotions, behaviors, and events, with a rigorous and understandable process.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.open_ended_questioning_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**5. Ability to Ask Open-Ended Questions**
- **Asking Meaningful and Precise Open-Ended Questions, Clear and Direct Expression**
  - **0 points:** The questions are not open-ended or are too vague, failing to effectively guide the client’s thinking or facilitate in-depth exploration. The questions are unclear or overly complex, causing confusion for the client.
  - **1 point:** The questions have some degree of openness and can guide the client’s thinking, but the phrasing may not be precise or fully clear, possibly requiring multiple attempts to prompt deeper reflection. The questions are relatively clear, but sometimes may seem somewhat complex or vague, requiring further explanation.
  - **2 points:** The questions are very precise and highly open-ended, effectively stimulating deep thinking from the client. The phrasing is simple, clear, and strongly guiding, allowing the client to easily understand the core of the question and quickly engage in thoughtful reflection.
  
- **Guiding the Client to Explore the Underlying Reasons for Their Inner Feelings, Thoughts, and Behaviors**
  - **0 points:** The questions do not effectively guide the client to explore the underlying reasons for their feelings, thoughts, or behaviors, lacking depth.
  - **1 point:** The questions somewhat guide the client in exploring their inner world, but the guidance may not be deep enough to fully stimulate self-exploration.
  - **2 points:** The questions profoundly guide the client to reflect on the deeper causes of their feelings, thoughts, and behaviors, promoting deep self-exploration and insight.
  
- **Stimulating the Client’s Self-Reflection and New Insights on the Issue**
  - **0 points:** The questions fail to stimulate self-reflection or generate new insights from the client, leaving the conversation at a superficial level.
  - **1 point:** The questions guide the client to reflect to some extent, but the insights are limited, and deeper thinking is not fully stimulated.
  - **2 points:** The questions deeply stimulate the client’s self-reflection, prompting new insights and helping the client see the issue from different perspectives.
  
- **Remaining Focused on the Core Topic**
  - **0 points:** The questions stray from the core topic, causing the conversation to lose focus or fail to effectively address the central issue.
  - **1 point:** Most of the questions stay focused on the core topic, but occasional deviations occur, requiring some redirection or adjustment.
  - **2 points:** The questions consistently focus on the core topic, maintaining relevance and ensuring the conversation stays centered on the main issue.
  
- **Flexibly Adjusting the Direction of Questions to Ensure Depth and Coherence in the Dialogue**
  - **0 points:** The questions are rigid and fail to adjust based on the conversation’s progress or the client’s responses, leading to insufficient depth or coherence in the dialogue.
  - **1 point:** The questions can be adjusted based on the conversation’s progress or the client’s responses, but the adjustments are not always flexible or timely, leading to occasional lapses in coherence.
  - **2 points:** The questions are highly flexible, adjusting dynamically to the flow of the conversation and client responses, ensuring the discussion remains deep and coherent throughout.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.positive_guidance_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**6. Positive Guidance Ability**
- **Consistently Guiding the Client in a Positive and Constructive Manner**
  - **0 points:** Fails to guide the client in a positive or constructive manner, may be overly negative or respond inappropriately, failing to motivate the client effectively.
  - **1 point:** In most cases, maintains a positive and constructive approach, but during discussions of difficult or negative emotions, may show some negativity or provide less in-depth responses.
  - **2 points:** Consistently guides the client in a positive and constructive manner, helping the client maintain hope and find solutions even in challenging situations, encouraging them to move forward actively.
  
- **Helping the Client Find Hope, Solutions, or New Perspectives Even During Difficult or Negative Emotions**
  - **0 points:** Fails to help the client during difficult or negative emotions, leaving the client feeling helpless or worsening their emotional state.
  - **1 point:** In some cases, helps the client find new perspectives or solutions, but may not fully motivate the client to explore hopeful directions or respond sufficiently to negative emotions.
  - **2 points:** Effectively helps the client find new perspectives, instills hope, and offers practical solutions or action steps during difficult or negative emotional discussions, helping the client regain confidence.
  
- **Skilled at Identifying the Client’s Resources and Strengths**
  - **0 points:** Fails to identify the client’s resources and strengths, not using these effectively to help the client resolve issues.
  - **1 point:** Identifies the client’s resources and strengths to some extent, but may not fully utilize them or may occasionally overlook key internal strengths.
  - **2 points:** Highly attuned to identifying the client’s resources and strengths, and effectively guides the client in leveraging these to overcome challenges and resolve issues.
  
- **Motivating the Client to Take Positive Action Toward Clear Goals**
  - **0 points:** Fails to motivate the client to take action, lacking clear goals or direction, leading the client to stagnate.
  - **1 point:** Motivates the client to take some action and move toward a goal, but may not fully clarify the goal or inspire sufficient motivation.
  - **2 points:** Continuously motivates the client to take positive action, helps them define clear goals, and creates actionable steps, ensuring steady progress toward those goals.
  
- **Providing Ongoing Encouragement and Support Throughout the Process**
  - **0 points:** Fails to provide ongoing encouragement and support, leading the client to feel isolated or lose motivation.
  - **1 point:** Provides encouragement and support, but may be intermittent, occasionally lacking sustained focus or feedback.
  - **2 points:** Provides continuous encouragement and support, consistently following up on the client’s progress and offering feedback, helping the client feel supported and motivated throughout the counseling process.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.continuous_dialogue_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**7. Continuous Dialogue Ability**
- **Maintaining a Smooth and Natural Conversation**
  - **0 points:** The conversation is not smooth or feels awkward, with noticeable pauses or difficulty continuing, lacking a natural flow of communication.
  - **1 point:** The conversation is generally smooth, but occasional pauses may occur or more guidance may be needed to maintain a natural rhythm and flow.
  - **2 points:** The conversation is very smooth and natural, maintaining a good rhythm throughout, with no pauses or awkward silences, allowing the discussion to flow easily.
  
- **Continuing the Conversation Through Precise Questions, Responses, and Feedback**
  - **0 points:** Questions or responses are not precise, failing to effectively continue the conversation, leading to interruptions or topic divergence.
  - **1 point:** Most questions, responses, and feedback effectively continue the conversation, but occasionally they may lack precision or be somewhat vague, affecting the flow and coherence.
  - **2 points:** Questions, responses, and feedback are all highly precise and well-timed, naturally extending the conversation and deepening the discussion.
  
- **Skilled at Capturing the Key Points in the Client’s Speech**
  - **0 points:** Fails to accurately capture the key points in the client’s speech, missing or misunderstanding the client’s core concerns or issues.
  - **1 point:** Able to capture the key points in the client’s speech most of the time, but occasionally may overlook certain details or nuances.
  - **2 points:** Highly attuned to capturing the key points in the client’s speech, accurately identifying and responding to the client’s core emotions, issues, or needs.
  
- **Responding Appropriately to the Client’s Emotions and Expression**
  - **0 points:** Fails to respond appropriately to the client’s emotions or expressions, with responses that feel stiff or mismatched to the emotional tone.
  - **1 point:** Generally responds appropriately to the client’s emotions, but may occasionally be slow to react or not fully align with the client’s emotional state.
  - **2 points:** Consistently responds with precision to the client’s emotions and expressions, providing empathetic and emotionally attuned responses that meet the client’s emotional needs.
  
- **Ensuring the Conversation Remains Deep and Coherent**
  - **0 points:** The conversation lacks coherence, with topics jumping around or stagnating, failing to maintain depth or meaningful discussion.
  - **1 point:** The conversation generally maintains coherence and depth, but occasionally experiences pauses or diverges from the topic, affecting the flow.
  - **2 points:** The conversation remains consistently deep and coherent, with a smooth development of topics, each point naturally connecting to the next, without any divergence or pauses.
  
- **Guiding the Client Flexibly and Subtly to Avoid Pauses or Awkward Silences**
  - **0 points:** Fails to guide the client flexibly, leading to pauses or silences, leaving the client confused or unsure of how to continue.
  - **1 point:** Able to flexibly guide the conversation most of the time, but occasional pauses or silences may occur, requiring further adjustments and guidance.
  - **2 points:** Skillfully and subtly guides the client, preventing any pauses or awkward silences, ensuring the conversation remains deep and productive throughout.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.context_summary_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**8. Context Summary Ability**
- **Accurately Extracting Key Points from the Conversation**
  - **0 points:** Fails to effectively extract the key points from the conversation, with responses that are too general or off-topic, lacking accurate capture of the client's emotions, thoughts, and expressions.
  - **1 point:** Able to extract some key points, but may overlook important details or fail to fully capture the client's core emotions and thoughts.
  - **2 points:** Accurately extracts all key points from the conversation, clearly identifying the client's primary emotions, thoughts, and expressions, and summarizing the core content of the discussion with precision.
  
- **Integrating the Client’s Emotions, Thoughts, and Experiences Organically**
  - **0 points:** Fails to effectively integrate the client’s emotions, thoughts, and experiences, resulting in disorganized or fragmented expressions that fail to help the client see the overall connections.
  - **1 point:** Able to integrate the client’s emotions, thoughts, and experiences to some extent, but the integration may lack cohesion or flow, and sometimes may not clearly show the internal connections between them.
  - **2 points:** Smoothly and organically integrates the client’s emotions, thoughts, and experiences, helping the client see the internal connections between various aspects and forming a clear, cohesive view of the overall situation.
  
- **Linking Current Discussion with Previous Content or the Larger Context During Summaries**
  - **0 points:** Fails to effectively link the current discussion with previous content or the overall context, lacking summary or continuity, and leaving the conversation feeling disjointed or shallow.
  - **1 point:** Able to make some connections between the current discussion and previous content during summaries, but may not fully demonstrate logical continuity or may lack tight integration between the points.
  - **2 points:** Skillfully links the current discussion with previous content or the larger context during summaries, clearly showing the logical connections and progression between points, and effectively demonstrating the development of the discussion.
  
- **Helping the Client Build Clear Logical Connections Between Different Topics**
  - **0 points:** Fails to help the client build connections between different topics, with a lack of logical flow between topics, leaving the client confused or unable to understand the overall coherence of the conversation.
  - **1 point:** Able to help the client establish some connections between different topics, but these connections may not be strong or clear, and some topics may feel loosely related or unclear.
  - **2 points:** Effectively helps the client build clear logical connections between different topics, ensuring that all points in the conversation are interrelated and easy for the client to understand in terms of their underlying connections.
  
- **Helping the Client Better Understand Their Motivations, Internal Conflicts, Action Directions, Inner Experiences, and Growth Process**
  - **0 points:** Fails to help the client gain a deep understanding of their motivations, internal conflicts, or action directions, lacking meaningful reflection or self-awareness.
  - **1 point:** Able to help the client understand their motivations, internal conflicts, or action directions to some extent, but fails to fully uncover deeper emotions or internal conflicts.
  - **2 points:** Deeply helps the client understand their motivations, internal conflicts, action directions, inner experiences, and growth process, sparking greater self-awareness and profound reflection on these aspects.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.resistance_handling_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**9. Resistance Handling Ability**
- **Keenly Identifying the Client's Defense Mechanisms and Resistance Behaviors**
  - **0 points:** Fails to identify or overlooks the client's defense mechanisms and resistance behaviors, missing emotional reactions or failing to detect the client’s protective responses.
  - **1 point:** Able to identify some of the client's defense mechanisms and resistance behaviors, but may not fully capture their complexity, and some behaviors may be overlooked or misinterpreted.
  - **2 points:** Very keenly identifies the client’s defense mechanisms and resistance behaviors, accurately capturing the client’s emotional responses and underlying psychological defenses.
  
- **Skillfully Guiding the Client to Face These Psychological Defenses, Avoiding Confrontation**
  - **0 points:** Fails to effectively guide the client in facing their defense mechanisms, which may lead to confrontation or exacerbate the client’s defensive emotions, hindering the progress of therapy.
  - **1 point:** Able to moderately guide the client to face their defense mechanisms, but may sometimes be too direct or slightly rigid, not fully avoiding confrontation or conflict.
  - **2 points:** Skillfully guides the client in facing their defensive behaviors, responding to the client’s defenses with flexibility and gentleness, ensuring the conversation does not provoke confrontation or resistance.
  
- **Using Gentle and Effective Strategies to Help the Client Gradually Lower Their Defenses**
  - **0 points:** Fails to effectively lower the client’s defenses, leaving the client in a state of resistance and unable to explore their emotions or issues deeply.
  - **1 point:** Able to somewhat reduce the client’s defenses using strategies, but the approach may not be flexible or gentle enough, failing to fully encourage the client to drop their defenses.
  - **2 points:** Gently and effectively uses strategies to help the client gradually lower their defenses, allowing the client to feel safe enough to gradually drop their defenses and explore themselves more deeply.
  
- **Helping the Client Deeply Explore Their Emotions and Underlying Issues While Maintaining Trust and Safety**
  - **0 points:** Fails to help the client explore their emotions and underlying issues in depth, lacking trust and safety, making the client feel uncomfortable or rejected.
  - **1 point:** Able to help the client explore their emotions and underlying issues to some extent, but may not fully maintain trust and safety, causing the client to feel misunderstood or uneasy at times.
  - **2 points:** Helps the client explore their emotions and underlying issues deeply while consistently maintaining trust and safety, ensuring the client feels accepted and understood throughout the process.
  
- **Transforming the Client’s Resistance into Opportunities for Growth Through Empathy and Understanding**
  - **0 points:** Fails to effectively transform the client’s resistance, leaving the resistance unresolved and hindering progress in therapy.
  - **1 point:** Able to somewhat transform the client’s resistance, but the transformation is not very noticeable or is slow, failing to fully facilitate the client’s growth.
  - **2 points:** Successfully transforms the client’s resistance into an opportunity for growth through empathy and understanding, facilitating smooth progress in therapy and helping the client gradually break through their self-defense mechanisms.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"

        self.humanized_expression_scoring_prompt = f"""system
You are a strict but fair expert in evaluating the quality of psychological counseling sessions. Your standards are high, yet you recognize that perfect counseling is nearly impossible. Your goal is to identify high-quality counseling dialogues and provide objective, professional assessments.

**Task:** Conduct a thorough quality evaluation of the provided psychological counseling dialogue based on the following scoring dimensions and criteria. For each dimension, offer detailed analysis and improvement suggestions. Finally, calculate and provide an overall score. Your evaluations should ensure that only approximately 10% of dialogues are rated as excellent, while the remaining 90% are deemed below standard. Specifically, only about 10% of dialogues should receive a score of 9 or above.

**Scoring Dimensions List** (High scores require near-perfect performance in most aspects):

**10. Humanized Expression Ability**
- **Using Clear and Understandable Everyday Language**
  - **0 points:** The language is unclear or overly formal, using complex or difficult-to-understand vocabulary and sentence structures, making it hard for the client to follow.
  - **1 point:** The language is generally clear, with simple expressions, but occasionally uses words or sentence structures that may be difficult to understand or feel unnatural.
  - **2 points:** The language is concise, conversational, and easy to understand, closely matching the client’s everyday language, ensuring the client can easily follow and engage in the conversation.

- **Expressing Ideas Fluently and Using Everyday Language**
  - **0 points:** The expression is stiff, disjointed, or lacks fluency, making it difficult to resonate with the client’s everyday language.
  - **1 point:** The expression is generally fluent, but occasionally the sentences are not natural or do not fully align with the client’s everyday language, leading to slight awkwardness.
  - **2 points:** The expression is very fluent and natural, completely aligned with the client’s everyday language, ensuring smooth and friendly conversation.

- **Demonstrating Warmth and Avoiding Difficult or Complex Vocabulary**
  - **0 points:** Lacks warmth, using difficult vocabulary or complex sentence structures that create a sense of distance and make it hard for the client to relate.
  - **1 point:** Generally shows warmth and avoids difficult language, but may sometimes use expressions that feel less friendly or not fully relatable to the client.
  - **2 points:** Very warm and approachable, using simple and direct language, always avoiding hard-to-understand or awkward words, ensuring the client feels accepted and understood.

- **Responding to the Client’s Emotions with a Warm and Sincere Tone**
  - **0 points:** Fails to effectively respond to the client’s emotions, using a cold or mechanical tone that lacks emotional resonance, making the client feel ignored or misunderstood.
  - **1 point:** Able to respond to the client’s emotions with warmth in most cases, but the tone may occasionally feel flat or insincere, not fully conveying emotional understanding.
  - **2 points:** Always responds to the client’s emotions with a warm, sincere tone, effectively conveying empathy and understanding, making the client feel accepted and supported.

- **Using Humanized Language to Increase Emotional Resonance and Impact**
  - **0 points:** The language lacks a human touch, failing to resonate with the client’s emotions and making the conversation feel distant or detached.
  - **1 point:** Uses some degree of humanized language to convey emotional resonance, but may not always be deep or emotionally moving enough.
  - **2 points:** Successfully uses sincere and humanized language, making the conversation more emotionally resonant and impactful, allowing the client to feel their emotions are fully understood and accepted.

Please evaluate the following psychological counseling dialogue based on the above scoring criteria and output the score in JSON format, precise to two decimal places:

[multi_dialog]

**Response Format:**
- **Analysis:** (Provide a thorough, accurate, and professional analysis explaining why this score was given. Extract specific examples from the original dialogue to support your analysis.)
- **Improvement Suggestions:** (Provide detailed, accurate, and professional improvement suggestions. For any inappropriate expressions used by the counselor, offer more professional and targeted alternative expressions.)

""" + "{'score':0-10}"
