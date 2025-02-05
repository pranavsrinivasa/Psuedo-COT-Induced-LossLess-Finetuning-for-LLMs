class CoTLLM:
    def __init__(self,model_finetuned,TTM_model,tokenizer):
        self.model = model_finetuned
        self.TTM_model = TTM_model
        self.tokenizer = tokenizer
        pass    
    def generate_output(self,text,max_cot = 1):
        buffer = []
        buffer.append(text)
        for i in range(max_cot):
            input_text = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Role: You are an Instructions Providing AI that generates tailored steps to solve the given question.

            Instructions:
            - Carefully analyze the provided question before generating steps.
            - Only generate **specific steps** that are relevant to solving this particular question.
            - Avoid using generic or repetitive steps such as "Identify key information" or "Verify the solution".
            - Focus on logical reasoning, calculations, or operations that are **directly necessary** to solve the question.
            - Give Exactly 4 steps.
            - Do not provide descriptions or explanations for the steps.
            - Only output the **step titles** relevant to the question at hand.
            - Follow the provided format:
            Step 1: [Tailored Step Title According to Question]
            Step 2: [Tailored Step Title According to Question]
            Step 3: [Tailored Step Title According to Question]
            Step 4: [Tailored Step Title According to Question]
            - Do not generate an answer to the question or hint at the solution.
            - Do not exceed 4 Steps
            Generate the steps based solely on the question below.
            <|eot_id|><|start_header_id|>user<|end_header_id|>"{text}"<|eot_id|>"""
            # internal_thought = self.generate_response(self.TTM_model, self.tokenizer, input_text,isTTM = True)
            internal_thought = self.generate_response(self.TTM_model, self.tokenizer, input_text,isTTM = True)
            internal_thought = self.extract_final_answer(internal_thought)
            internal_thought = f"Internal_thought{i+1}:"+"\n"+internal_thought
            buffer.append(internal_thought)
            prompt = '\n'.join(buffer)
            final_input = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Role: You are a highly intelligent AI assistant specializing in mathematics and coding. Your expertise includes solving complex math problems, writing and debugging code, explaining mathematical concepts, and providing optimized solutions for coding challenges. When presented with a question or problem, you will: 1. Analyze the problem carefully. 2. Provide clear and concise explanations for your reasoning. 3. Offer step-by-step solutions for math and coding problems. 4. Generate clean, efficient, and well-commented code for programming tasks. You are expected to be accurate, logical, and detailed in your responses.
            Instruction:
            - Use the internal_thought to guide yourself to a correct answer and verify that it is correct before responding to the user.
            - Final output needs to be an answer for the question.
            - The last sentence needs to be the correct option for the question.
            - Provide the index of the correct option
            - Always provide the correct option number at the end
            - Follow the strictly the Structure of output:
                Explanation : Elaborate on steps in internal_thought provided
                Answer : Correct Answer
                Option : Correct Option number for the correct answer in the choices
            - Do not deviate from the format mentioned above
            - Option can only be any one value in 0,1,2,3 and should only be the option number
            - Do not hallucinate
            - Do not deviate from the instructions
            Example:
            Question : What is 1 + 2 ?
            Choices:
            0) 3
            1) 1
            2) 2
            3) 4
            Explanation : 1 + 2 adds to 3
            Answer : the answer is 3
            Option : 0
            <|end_header_id|>
            <|start_header_id|>user<|end_header_id|>"{prompt}"<|eot_id|>"""
            final_output = self.generate_response(self.model, self.tokenizer, final_input)
            final_output = f"{i+1}th Output:\n" + final_output
            buffer.append(final_output)
        return final_output
    def extract_final_answer(self,output: str) -> str:
        # Assuming "Assistant:" precedes the answer
        if "assistant" in output:
            temp = output.split("assistant")[-1].strip()
            res = temp.replace('assistant','')
            return res
        return output.strip()
    def generate_response(self,model, tokenizer, input_text, isTTM = False):
        max_length = 512
        input_ids = tokenizer(input_text, return_tensors="pt")
        if not isTTM:
            outputs = model.generate(input_ids['input_ids'], max_new_tokens=2048,temperature = 0.6, do_sample = True, top_k = 50,top_p = 0.95)
        else:
            outputs = model.generate(input_ids['input_ids'], max_new_tokens=max_length, num_beams = 2, early_stopping = True)
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return res