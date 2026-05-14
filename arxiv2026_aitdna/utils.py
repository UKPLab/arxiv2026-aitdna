import os
import json
from openai import OpenAI
from google import genai
import requests


def use_openai_client(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    """
    Send a query using OpenAI client

    :param self:
    :param model: model name
    :type model: str
    :param client: client to use
    :type client: OpenAI
    :param prompt: user prompt
    :type prompt: str
    :return: model answer
    :rtype: str
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=temperature
    )
    text = response.choices[0].message.content
    return text


def use_ollama_client(url: str, model: str, prompt: str, temperature: float) -> str:
    """
    Send a query using OpenAI client

    :param self:
    :param model: model name
    :type model: str
    :param url: server to connect to
    :type url: str
    :param prompt: user prompt
    :type prompt: str
    :return: model answer
    :rtype: str
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    result = requests.post(url, data=json.dumps(payload), timeout=30)

    return result.json()["response"]


def send_request(model: str, prompt: str, temperature: float) -> dict[str, bool | str]:
    """
    Sends a request to an LLM and returns the answer.

    :param self:
    :param model: model name
    :type model: str
    :param prompt: prompt
    :type prompt: str
    :return: Model answer and whether the request was successful
    :rtype: dict[str, bool | str]
    """
    if "deepseek" in model:
        client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        return use_openai_client(client, model, prompt, temperature)
    elif "gpt" in model:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return use_openai_client(client, model, prompt, temperature)

    elif "gemini" in model:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature
            )
        )
        return response.text

    elif "llama" in model:
        url = os.environ.get("OLLAMA_API")
        return use_ollama_client(url=url, model=model, prompt=prompt, temperature=temperature)
    else:
        url = os.environ.get("UKP_LLM_API")
        return use_ollama_client(url=url, model=model, prompt=prompt, temperature=temperature)

