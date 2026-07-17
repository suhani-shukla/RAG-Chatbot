"""
Part 7: Connect the Retriever to an LLM
-------------------------------------------
Provider-agnostic: set LLM_PROVIDER (env variable) to switch between
Gemini, OpenAI, and a local model served through Ollama, without
changing any other file.

Options:
  "gemini" -> uses the Google Gemini API. Requires GEMINI_API_KEY env variable.
  "openai" -> uses the OpenAI API. Requires OPENAI_API_KEY env variable.
  "ollama" -> uses a local model via Ollama. No API key needed, but
              requires Ollama installed and running locally.

API keys and other config are loaded from a .env file in the project
root (via python-dotenv) if present, falling back to real environment
variables otherwise.

Usage from other modules:
    from llm import generate_answer
    answer = generate_answer(prompt)
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present, into os.environ

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini", "openai", or "ollama"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def _generate_gemini(prompt: str) -> str:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey "
            "and add it to your .env file as: GEMINI_API_KEY=your-key-here"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def _generate_openai(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file as: "
            "OPENAI_API_KEY=your-key-here"
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _generate_ollama(prompt: str) -> str:
    import ollama

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        raise ConnectionError(
            f"Could not reach Ollama. Is it installed and running? "
            f"(https://ollama.com). Have you pulled the model with "
            f"'ollama pull {OLLAMA_MODEL}'? Original error: {e}"
        )


def generate_answer(prompt: str) -> str:
    """Route the prompt to whichever provider is configured."""
    if LLM_PROVIDER == "gemini":
        return _generate_gemini(prompt)
    elif LLM_PROVIDER == "openai":
        return _generate_openai(prompt)
    elif LLM_PROVIDER == "ollama":
        return _generate_ollama(prompt)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Use 'gemini', 'openai', or 'ollama'."
        )


if __name__ == "__main__":
    print(f"Testing LLM provider: {LLM_PROVIDER}")
    test_prompt = "In one sentence, what is Retrieval-Augmented Generation?"
    try:
        print(generate_answer(test_prompt))
    except Exception as e:
        print(f"(Skipping live test — provider not ready yet: {e})")