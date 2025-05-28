import logging
import json

from bedrock_controller import BedrockController
from parser import parse_input

logger = logging.Logger(__name__)

brc = BedrockController()


def lambda_handler(event, context):
    logger.info("Received event: " + json.dumps(event, indent=2))

    input_text = None
    thread_id = None

    try:
        if "body" in event:
            try:
                body = json.loads(event["body"])
                input_json = body.get("input")
                thread_id = body.get("thread_id")
                input_text = parse_input(input_json)

            except json.JSONDecodeError:
                logger.error("Failed to parse request body as JSON")

        # If no input found in body, try to find it directly in the event
        if not input_text:
            input_json = event.get("input")
            thread_id = event.get("thread_id")
            input_text = parse_input(input_json)

        logger.info(f"Received input: {input_text}")

        if not input_text or not thread_id:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing input parameter."}),
            }

        llm_response = brc.converse(input_text, thread_id)

        logger.info("LLM called successfully")

        # Format response based on whether it's from Lambda URL or direct
        if "body" in event:
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"Answer": llm_response}),
            }
        else:  # Direct Lambda invocation
            return {"Answer": llm_response}

    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        error_response = {"error": str(e)}

        if "body" in event:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(error_response),
            }
        else:  # Direct Lambda invocation
            return error_response
