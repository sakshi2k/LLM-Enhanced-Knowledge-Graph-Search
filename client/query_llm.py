import os
from openai import OpenAI
from prompt import MASTER_PROMPT
from sample_input_jsons_for_dev.sample_data_payload import SAMPLE_DATA


client = OpenAI(
    base_url = "https://router.huggingface.co/v1",
    api_key = os.getenv("HF_TOKEN_KEY")
)

def make_llm_call_hf(query:str, speeches_list:str):
    print("Requesting LLM response")

    prompt = MASTER_PROMPT.replace("{{USER_QUERY}}", query).replace("{{SPEECHES_LIST}}", str(speeches_list))
    print("Prompt constructed :: ", prompt)


    response = client.responses.create(
        model="openai/gpt-oss-120b:groq",
        input=prompt,
    )

    print("LLM Response obtained")
    print(response)
    print("LLM Response text")
    print(response.output_text)
    return response.output_text

if __name__ == "__main__":

    make_llm_call_hf("What changed in ECB messaging on growth between the years 2022 and 2025", SAMPLE_DATA['results'])
    print("Done")