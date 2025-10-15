import os, json, psutil, re
from contextlib import redirect_stderr
from llama_cpp import Llama

def call_api(prompt, options, context):
    """Run harm classification via `label_conversation` and return results for Promptfoo.

    Args:
        prompt (str): Instruction guiding the model.
        options (dict): Optional parameters (unused).
        context (dict): Contains the conversation transcript.

    Returns:
        dict: {"output": "<JSON result>"} or {"error": "<error message>"}.
    """
    transcript = context.get("vars", {}).get("transcript", "")

    try:
        reason = label_conversation(structured_transcript=transcript, summary_prompt=prompt)
        # Wrap in the 'output' field as required by Promptfoo. JSON-encode dicts.
        return {"output": json.dumps({"response": reason}, ensure_ascii=False)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

def label_conversation(structured_transcript, summary_prompt):
    """Use a local Llama model to classify if a conversation is harmful.

    Args:
        structured_transcript (str): Conversation text.
        summary_prompt (str): Model instruction or classification guide.

    Returns:
        tuple: (label, reason) where label ∈ {0,1} and reason is a short explanation.
    """
    available_gb = psutil.virtual_memory().available / (1024**3)
    ctx = 2048 if available_gb > 8 else 1024 if available_gb > 4 else 512

    null = open(os.devnull, "w")
    with redirect_stderr(null):
        llm = Llama(
            model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            n_ctx=ctx,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )

    to_summarise = f"{summary_prompt}\n\n Now summarise the conversation :\n{structured_transcript}"
    summary_out = llm(to_summarise, max_tokens=128, temperature=0.5)
    raw_response = summary_out['choices'][0]['text'].strip()
    label, reason = extract_label_reason(raw_response)
    llm.reset() 
    return label, reason

def extract_label_reason(response):
    """Parse model output to extract harm label and reason.

    Args:
        response (str): Raw model response text.

    Returns:
        tuple: (label, reason) with defaults if extraction fails.
    """
    try:
        # Extract all JSON-like substrings in the response
        json_objects = re.findall(r'\{[^{}]*"label"\s*:\s*\d[^{}]*"reason"\s*:\s*"[^"]*"[^{}]*\}', response)

        for obj in reversed(json_objects):  # Check from last to first
            try:
                parsed = json.loads(obj)
                return int(parsed.get("label", 0)), parsed.get("reason", "")
            except json.JSONDecodeError:
                continue

        # As fallback, try full string if it is a clean JSON
        parsed = json.loads(response)
        return int(parsed.get("label", 0)), parsed.get("reason", "")

    except Exception as e:
        print("Failed to extract label/reason:", e)
        return 0, "Could not extract reason."