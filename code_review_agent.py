"""
Author: Alex Punnen
Status:  Demo
This is a simple python based Code Review Agent flow using OpenAI LLM APIs amd Model Context Protocl based client
Design patterns like Command Pattern are used along with for loops to stucture flow and response as we need

"""
import git_utils
from fastmcp import Client
from openai import OpenAI
from dotenv import load_dotenv
import logging as log
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import sys
import inspect
import yaml
from typing import Any

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(f"Parent directory: {parentdir}")
# add the parent directory to the system path
sys.path.append(parentdir)
from nmagents.command import CallLLM, ToolCall, ToolList
from nmagents.utils import parse_json_response_with_repair,execute_step_tools
# configure logging
from pathlib import Path

__author__ = "Alex Punnen"
__version__ = "1.0.0"
__email__ = "alexcpn@gmail.com"


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
os.makedirs("./logs", exist_ok=True)
time_hash = datetime.now().strftime("%Y%m%d%H%M%S")
outfile = "./logs/out_" + time_hash + "_" + ".log"
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  #
    # format="[%(levelname)s] %(message)s",  # dont need timing
    handlers=[log.FileHandler(outfile), log.StreamHandler()],
    force=True,
)
# Load the .env file and get the API key
load_dotenv()

# https://platform.openai.com/api-keys add this to your .env file
api_key = os.getenv("OPENAI_API_KEY")
MAX_CONTEXT_LENGTH = 16385
MAX_RETRIES = 5
# USD  # https://platform.openai.com/docs/pricing for gpt-4.1-nano
COST_PER_TOKEN_INPUT = 0.10/10e6
COST_PER_TOKEN_OUTPUT = .40/10e6  # USD
AST_MCP_SERVER_URL = os.getenv(
    "CODE_AST_MCP_SERVER_URL",
    "http://127.0.0.1:7860/mcp/",
)

if AST_MCP_SERVER_URL and not AST_MCP_SERVER_URL.endswith("/"):
    AST_MCP_SERVER_URL = AST_MCP_SERVER_URL + "/"

# Initialize OpenAI client with OpenAI's official base URL
MODEL_NAME = "gpt-4.1-nano"
#MODEL_NAME = "gpt-5-nano"
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"
)

# ollama client
# MODEL_NAME= "phi3.5" 
# openai_client = OpenAI(
#     api_key="sk-local",
#     base_url="http://localhost:11434/v1"
#  )

# vllm client
# MODEL_NAME= "gemma"
# openai_client = OpenAI(
#     api_key="sk-local",
#     base_url="http://localhost:8080/v1"
# )


FALLBACK_MODEL_NAME = os.getenv("JSON_REPAIR_MODEL", "gpt-4.1-nano" )
FALLBACK_MAX_BUDGET = float(os.getenv("JSON_REPAIR_MAX_BUDGET", "0.2"))


app = FastAPI()

  
# add current directory path

TEMPLATE_PATH = Path(__file__).parent / "prompts/code_review_prompts.txt"



def load_prompt(**placeholders) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    default_values = {
        "arch_notes_or_empty": "",
        "guidelines_list_or_link": "",
        "threat_model_or_empty": "",
        "perf_slos_or_empty": "",
        "tool_outputs": "",
        "diff_or_code_block": "",
    }
    merged = {**default_values, **placeholders}
    for key, value in merged.items():
        value_str = str(value)
        template = template.replace(f"{{{{{key}}}}}", value_str)
        template = template.replace(f"{{{key}}}", value_str)
    return template

@app.get("/review")
async def review(repo_url: str, pr_number: int):
    log.info(f"Received review request for {repo_url} PR #{pr_number}")
    try:
        review_comment = await main(repo_url, pr_number)
    except Exception as exc:
        log.exception("Error executing review")
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)
    return JSONResponse(content={"status": "ok", "review_comment": review_comment or "No review comment produced."})

async def main(repo_url, pr_number):

    # Example: get the diff for a specific PR
    print(f"Code review for PR #{pr_number} from {repo_url}...")

    # ------------------------------------------------
    #  Command to Call the LLM with a budget ( 0.5 Dollars)
    call_llm_command = CallLLM(openai_client, "Call the LLM with the given context",
                               MODEL_NAME, COST_PER_TOKEN_INPUT, COST_PER_TOKEN_OUTPUT, 0.5)
    repair_llm_command =   repair_llm_command = CallLLM(
        openai_client,
        "Repair invalid YAML responses",
        FALLBACK_MODEL_NAME,
        COST_PER_TOKEN_INPUT,
        COST_PER_TOKEN_OUTPUT,
        FALLBACK_MAX_BUDGET,
    )

    sample_step_schema_file = "schemas/steps_schema.json"
    log.info(f"Using step schema file: {sample_step_schema_file}")
    with open(sample_step_schema_file, "r", encoding="utf-8") as f:
        step_schema_content = f.read()
        

    # this this the MCP client invoking the tool - the code review MCP server
    async with Client(AST_MCP_SERVER_URL) as ast_tool_client:

        ast_tool_call_command = ToolCall(
            ast_tool_client, "Call the tool with the given method and params")
        ast_tool_list_command = ToolList(ast_tool_client, "List the available tools")

        tool_schemas_content = await ast_tool_list_command.execute(None)
        log.info(f"AST Tool schema: {tool_schemas_content}")

        file_diffs = git_utils.get_pr_diff_url(repo_url, pr_number)
        step_execution_results: list[dict[str, Any]] = []

        main_context = f""" Your task today is Code Reivew. You are given the following '{pr_number}' to review from the repo '{repo_url}' 
        You have to first come up with a plan to review the code changes in the PR as a series of steps.
        Write the plan as per the following step schema: {step_schema_content}
        Make sure to follow the step schema format exactly  and output only JSON """
        log.info("-"*80)
        log.info(f"Generating code review plan for PR #{pr_number} from {repo_url}")
        context = main_context
        stepcount = 0
        for file_path, diff in file_diffs.items():
            stepcount += 1
            log.info("-"*80)
            
            #------------------------------------------------
            # Prompt 1: Generate a plan for reviewing this file diff
            #------------------------------------------------
            
            context = main_context + f" Here is the file diff for {file_path}:\n{diff} for review\n" + \
                f"You have access to the following MCP tools to help you with your code review: {tool_schemas_content}"
            response = call_llm_command.execute(context)
            # log.info the response
            log.info(f"LLM response for step{stepcount} received")
            # parse the json response to check if its a plan or final review
            response_data, _ = parse_json_response_with_repair(
                response_text=response or "",
                schema_hint=step_schema_content,
                repair_command=repair_llm_command,
                context_label="plan",
            )
            with open(f"./logs/step_{stepcount}_{time_hash}.yaml", "w", encoding="utf-8") as f:
                yaml.dump(response_data, f)
            #------------------------------------------------
            # Now go through the steps for this file diff and execute it and
            #------------------------------------------------
            steps = response_data.get("steps", [])
            summary = response_data.get("summary", "")
            log.info(f"Generated plan summary: for step {stepcount} {summary}" )
            for index, step in enumerate(steps, start=1):
                name = step.get("name", "<unnamed>")
                step_description = step.get("description", "")
                log.info(f"Step {index}: {name}")
                log.info(f"Description: {step_description}")
                #------------------------------------------------
                # check if there are any tools to be executed for this step
                #------------------------------------------------
                tools = step.get("tools", [])
                 # execute the tools for this step  
                if tools is not None and len(tools) > 0:
                    log.info(f"Executing tools for step {name}: {tools}")
                    tool_outputs = await execute_step_tools(step, ast_tool_call_command)
                    for output_index, output in enumerate(tool_outputs, start=1):
                        log.info("Tool result %s for step %s: %s",
                                output_index, name, output)
                        tool_result_context = load_prompt(repo_name=repo_url, brief_change_summary=step_description,
                                                    diff_or_code_block=diff, tool_outputs=output)
                        # add this to the step context tool_results
                        step["tool_results"] = tool_result_context
                #------------------------------------------------
                # Prompt 2: Execute the step with the given context and any tool results
                #------------------------------------------------
                step_context = load_prompt(repo_name=repo_url, brief_change_summary=step_description,
                                           diff_or_code_block=diff, tool_outputs=step.get("tool_results", ""))
                log.info(f"Executing step {name} with context length {len(step_context)}")
                step_response = call_llm_command.execute(step_context)
                log.info(f"LLM response for step {name} received")
                response_data, _ = parse_json_response_with_repair(
                    response_text=step_response or "",
                    schema_hint="",
                    repair_command=repair_llm_command,
                    context_label=f"step {name}",
                        
                )
                with open(f"./logs/step_{stepcount}_{name}_done_{time_hash}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(response_data, f)
                step_execution_results.append(
                    {
                        "file_path": file_path,
                        "step_name": name,
                        "description": step_description,
                        "result": response_data,
                    }
                )
 
     
    call_llm_command.get_total_cost()
    return  context

