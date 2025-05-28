import logging
from typing import List

import boto3

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class BedrockController:
    """Controls information flow between user, AWS Bedrock and Knowledge Base."""

    SYSTEM_TEMPLATE = """
    You are a helpful assistant who answers user questions and bases his responses on the given context. 
    Be brief and keep your answers short and to the point. 
    Do NOT make bulleted and numbered lists when explaining something. You can write Python code.
    If the provided context doesn't contain information relevant to the user question, say "Cannot be answered based on the provided text."
    Don't be too enthusiastic.

    Contexts:
    <context>
    {context}
    </context>
    """

    QUERY_TRANSFORMER_SYSTEM_PROMPT = """
    You are a search query generator. Your ONLY function is to output 2-10 words that will be used as search terms.
    
    CRITICAL RULES:
    1. ONLY output keywords - NO explanations
    2. MAXIMUM 10 words total
    3. NO sentences
    4. NO punctuation except spaces
    5. NEVER say "I" or use first person
    6. NEVER explain your reasoning
    
    AUTOREGRESSIVE CHECK: If what you're writing would exceed 10 words or starts explaining, STOP IMMEDIATELY.
    
    EXAMPLES OF WRONG OUTPUTS:
    - "Runnable lambdas in Langchain offer several benefits including..."
    - "I'll create a search query for Langchain runnable lambdas"
    - "Here are some good keywords for your search:"
    
    EXAMPLES OF CORRECT OUTPUTS:
    - "langchain runnable lambdas"
    - "vector database integration langchain"
    - "langchain RAG implementation guide"
    """

    def __init__(self):
        # LLM Hyperparameters
        self.model_id = "amazon.nova-micro-v1:0"
        self.query_rewriter_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.answer_gen_inference_config = {
            "temperature": 0.1,
            "maxTokens": 500,
            "topP": 0.9,
        }
        self.q_rewrite_inference_config = {
            "temperature": 0.01,
            "maxTokens": 20,
            "topP": 0.1,
        }  # Inference configuration for low creativity

        self.bedrock_client = boto3.client(service_name="bedrock-runtime")
        self.kb_client = boto3.client("bedrock-agent-runtime")
        self.kb_id = "2OHLYXDVLT"
        self.message_histories = {}

    def _invoke_llm(self, retrieved_context, thread_id):
        """Queries a Bedrock LLM with contexts and chat history."""

        system_prompt_with_context = BedrockController.SYSTEM_TEMPLATE.format(
            context=[cont for cont in retrieved_context]
        )

        logger.info("Generating message with model %s", self.model_id)

        response = self.bedrock_client.converse(
            modelId=self.model_id,
            messages=self.message_histories[thread_id],
            system=[{"text": system_prompt_with_context}],
            inferenceConfig=self.answer_gen_inference_config,
            # additionalModelRequestFields=additional_model_fields
        )
        logger.info("LLM response generated.")

        token_usage = response["usage"]
        logger.info("Input tokens: %s", token_usage["inputTokens"])
        logger.info("Output tokens: %s", token_usage["outputTokens"])
        logger.info("Total tokens: %s", token_usage["totalTokens"])
        logger.info("Stop reason: %s", response["stopReason"])

        return response

    def _retrieve(self, query_term: str, top_k: int = 5) -> List:
        """Queries a predefined AWS Knowledge Base."""
        context = self.kb_client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                }
            },
            retrievalQuery={"text": query_term},
        )
        context = [chunk["content"]["text"] for chunk in context["retrievalResults"]]

        return context

    def converse(self, user_input: str, thread_id: str) -> str:
        """Controls Query Rewriting, Retrieval, Chat History management 
        and LLM inference."""
        if thread_id not in self.message_histories:
            self.message_histories[thread_id] = [
                {"role": "user", "content": [{"text": user_input}]}
            ]
        else:
            self.message_histories[thread_id].append(
                {"role": "user", "content": [{"text": user_input}]}
            )
        cur_thread = self.message_histories[thread_id]

        query_term = self._transform_query(thread_id)
        logger.info("Query rewriting done, with result of: " + query_term)

        context = self._retrieve(query_term)
        logger.info("Contexts retrieved: " + str(context))

        response = self._invoke_llm(context, thread_id)
        logger.info("LLM response: " + str(response))

        cur_thread.append(response["output"]["message"])

        # Forgetting
        if len(cur_thread) > 20:
            while cur_thread[0]["role"] != "user" and len(cur_thread) > 20:
                logger.info("Message history truncated.")
                cur_thread.pop(0)

        return response["output"]["message"]["content"][0]["text"]

    def _transform_query(self, thread_id: str) -> str:
        """
        Transform user query into search terms for retrieval.
        """
        query = self.bedrock_client.converse(
            modelId=self.query_rewriter_id,
            messages=self.message_histories[thread_id],
            system=[{"text": BedrockController.QUERY_TRANSFORMER_SYSTEM_PROMPT}],
            inferenceConfig=self.q_rewrite_inference_config,
        )

        return query["output"]["message"]["content"][0]["text"]
