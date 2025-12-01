---

title: Code Review Agent based on MCP
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: Dockerfile
pinned: false
---

# A Code Review Agent in Pure Python - No Frameworks

It uses the new Model Context Protocol exposed by [code_ast_mcp_server](code_ast_mcp_server) to get more context regarding functions, callees using MCP

This is added as an additonal context, along with the Merge request or Pull reuqest

### Using the Code Review MCP Server for a Code Review Automatiom

 The code for the workflow is in [code_review_agent.py](code_review_agent.py)

 It takes in a Github Repo URL, for exmaple '<https://github.com/huggingface/accelerate>' and the PR number say [3321](https://github.com/huggingface/accelerate/pull/3321) to review

 It gets the [diff of the files involved](https://patch-diff.githubusercontent.com/raw/huggingface/accelerate/pull/3321.diff) and used the MCP Server to find more details about the functions in the PR and their refrences.

 It does error handling and max retries as specified.

 It keeps track of LLM invocation and generation costs and keeps the workflow within specified budget

### Direct Invocaiton

```
cd code-review-agent
CODE_AST_MCP_SERVER_URL=http://127.0.0.1:7860/mcp/ uv run uvicorn code_review_agent:app --host 0.0.0.0 --port 8860
```

## Client

You can test with a python client like below

```
python3 client.py --repo-url https://github.com/huggingface/accelerate --pr-number 3321
```

You can expose the above vi Ngrok `ngrok http http://localhost:8860` or local.run `ssh -R 80:localhost:8860 localhost.run

### Integration to  Github Repo Webhook

Build the Docker and Run it in a server or via ngrok; and the the IP/URL to the  GitHub/Gitlab Webook

(Note 7860 is the port for HuggingFace Spaces)

```
docker build -t codereview-agent .

docker run -it --rm -p 7860:7860 codereview-agent
```

Example of the Webhook setting

![web hook](https://i.postimg.cc/LXcjtpv5/image.png)

### Schema-Driven Task Execution

- Use `task_schema.yaml` to describe available tasks and tools, then load it with `SchemaAgent`.
- Run `python3 schema_agent_demo.py --task fetch_pr_diffs --param repo_url=https://github.com/huggingface/accelerate --param pr_number=3321 --pretty` to execute the sample task.
- The agent casts inputs to the types declared in the schema and returns outputs following the same shape.
- Extend the schema with additional tools and tasks to auto-register new capabilities without hand-writing glue code.
- Inline `language: python` implementations are supported right inside the YAML, so you can ship small helper functions (like iterating file diffs) without creating extra modules.

### Output for one of the diff files using `gpt-4.1-nano` LLM and this MCP Server

 ```
 2025-05-20 17:03:47,352 [INFO] --------------------------------------------------------------------------------
2025-05-20 17:03:47,352 [INFO] Review diff for src/accelerate/utils/modeling.py
2025-05-20 17:03:47,352 [INFO] Total input tokens used: 0 Total output tokens generated: 0
2025-05-20 17:03:47,352 [INFO] Total cost: 0.0 
2025-05-20 17:03:49,652 [INFO] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-20 17:03:49,656 [INFO] LLM usage: CompletionUsage(completion_tokens=51, prompt_tokens=2338, total_tokens=2389, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))
2025-05-20 17:03:49,656 [INFO] LLM response: {
  "method": "get_function_context_for_project_mcp",
  "params": {
    "function_name": "infer_auto_device_map",
    "github_repo": "https://github.com/huggingface/accelerate"
  }
}
2025-05-20 17:03:49,777 [INFO] Total input tokens used: 2338 Total output tokens generated: 51
2025-05-20 17:03:49,777 [INFO] Total cost: 2.542e-05 
2025-05-20 17:03:51,501 [INFO] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-20 17:03:51,502 [INFO] LLM usage: CompletionUsage(completion_tokens=51, prompt_tokens=2392, total_tokens=2443, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=2176))
2025-05-20 17:03:51,502 [INFO] LLM response: {
  "method": "get_function_context_for_project_mcp",
  "params": {
    "function_name": "infer_auto_device_map",
    "github_repo": "https://github.com/huggingface/accelerate"
  }
}
2025-05-20 17:03:51,504 [INFO] Total input tokens used: 4730 Total output tokens generated: 102
2025-05-20 17:03:51,504 [INFO] Total cost: 5.138e-05 
2025-05-20 17:03:58,310 [INFO] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-05-20 17:03:58,311 [INFO] LLM usage: CompletionUsage(completion_tokens=344, prompt_tokens=2447, total_tokens=2791, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=2304))
2025-05-20 17:03:58,311 [INFO] LLM response: DONE: 

The `infer_auto_device_map` function has been significantly extended with an optional `reserve_max_layer` parameter, which, when set to `False`, disables the logic that reserves the maximum layer size for the main devices. This change can improve memory efficiency in scenarios with multiple GPUs or offloading, by preventing unnecessary reservation when offloading layers to CPU or disk.

Key points in the modification:
- Added `reserve_max_layer` parameter with default `True` to the function signature.
- Conditional checks around `get_max_layer_size` calls now depend on this parameter.
- When `reserve_max_layer` is `False`, the initial maximum layer size is set to 0, and subsequent updates only occur if needed.
- The logic that reduces `current_max_size` by `max_layer_size` is conditioned on `reserve_max_layer`.
- At the end, if `reserve_max_layer` is `False`, the function attempts to rerun itself with `reserve_max_layer=True` when offloading is detected, ensuring proper memory reservation in offloading scenarios.

Overall, the code provides flexible control over maximum layer size reservation, which can be beneficial for different deployment configurations. It maintains backward compatibility due to the default `True` setting.

Suggested improvements:
- Consider adding explicit warnings or documentation comments explaining when to disable reservation (`reserve_max_layer=False`) for advanced users.
- Ensure that the recursive call with `reserve_max_layer=True` won't cause unintended side effects or infinite loops—though the current logic seems safe given the condition.

No major structural issues detected. The code is clear with proper conditional handling around `reserve_max_layer`.

Would you like me to check the other functions used here or provide more detailed suggestions?
2025-05-20 17:03:58,311 [INFO] LLM finished the code review
```
