def basic_response_filter(response:str) -> str:
    response = response.strip('\n ')
    response = response.replace("<think>\n</think>", "")
    return response
